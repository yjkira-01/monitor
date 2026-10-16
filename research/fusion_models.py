"""Development fusion baselines and conditional residual-utility hypotheses.

These implementations are local controlled adaptations, not official paper
reproductions. All use the same RGB/point encoders and explicit missing inputs.
"""

import torch
from torch import nn
from torch.nn import functional as F

FUSION_METHODS = ("concat", "late_average", "quality_gate", "precision", "joint_attention",
                 "joint_gate", "utility_gate", "utility_global", "utility_nobone", "utility_random")
METHODS = ("rgb_single", "rgb_matched", "radar_single") + FUSION_METHODS
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def utility_targets(video, candidate, target, margin_mm=1., temperature_mm=10.):
    """Training-only online utility labels; detach avoids altering label errors.

    The comparison uses the UNGATED candidate. These moving labels are not
    cross-fitted or calibrated generalization-risk targets.
    """
    with torch.no_grad():
        video_error = (video - target).norm(dim=-1) * 1000
        candidate_error = (candidate - target).norm(dim=-1) * 1000
        return torch.sigmoid((video_error - candidate_error - margin_mm) / temperature_mm)


def bone_disagreement(video, candidate):
    value = torch.zeros_like(video[..., 0])
    degree = torch.zeros(17, device=video.device, dtype=video.dtype)
    for a, b in BONES:
        discrepancy = ((video[:, a] - video[:, b]).norm(dim=-1) -
                       (candidate[:, a] - candidate[:, b]).norm(dim=-1)).abs()
        value[:, a] = value[:, a] + discrepancy
        value[:, b] = value[:, b] + discrepancy
        degree[a] += 1
        degree[b] += 1
    return value / degree.clamp_min(1)


class PointFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())

    def forward(self, points, mask):
        encoded = self.encoder(torch.where(mask[..., None], points, 0.))
        mean = (encoded * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        maximum = encoded.masked_fill(~mask[..., None], torch.finfo(encoded.dtype).min).max(1).values
        maximum = torch.where(mask.any(1)[:, None], maximum, 0.)
        return encoded, torch.cat((mean, maximum), -1)


def pose_head(input_size):
    head = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 51))
    nn.init.normal_(head[-1].weight, std=.001)
    nn.init.zeros_(head[-1].bias)
    return head


class FusionModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown fusion method")
        self.method = method
        self.register_buffer("target_template_m", torch.zeros(17, 3))
        if method in ("rgb_single", "rgb_matched"):
            width = 245 if method == "rgb_matched" else 128
            self.rgb_encoder = nn.Sequential(nn.Linear(51, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            self.video_head = pose_head(width)
            return
        if method == "radar_single":
            self.point_encoder, self.radar_head = PointFeatures(), pose_head(128)
            return
        self.rgb_encoder = nn.Sequential(nn.Linear(51, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.point_encoder = PointFeatures()
        self.video_head, self.radar_head = pose_head(128), pose_head(128)
        if method == "concat":
            self.concat_head = pose_head(256)
        if method in ("quality_gate", "joint_gate", "utility_gate", "utility_global", "utility_nobone", "utility_random"):
            self.gate = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
        if method == "precision":
            self.video_variance = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 17))
            self.radar_variance = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 17))
        if method.startswith("utility") or method in ("joint_attention", "joint_gate"):
            self.video_projection = nn.Linear(128, 64)
            self.joint_embedding = nn.Parameter(torch.randn(17, 16) * .02)
            self.query = nn.Linear(80, 64)
            self.key, self.value = nn.Linear(64, 64), nn.Linear(64, 64)
            self.residual = nn.Sequential(nn.Linear(144, 64), nn.ReLU(), nn.Linear(64, 3))
            nn.init.normal_(self.residual[-1].weight, std=.001)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, rgb, points, mask, confidence):
        if rgb.shape != (len(rgb), 51) or points.shape != (len(rgb), 196, 5) or mask.shape != points.shape[:-1] or confidence.shape != (len(rgb), 17):
            raise ValueError("Unexpected fusion input shapes")
        if mask.dtype != torch.bool or not torch.isfinite(rgb).all() or not torch.isfinite(points[mask]).all() or not torch.isfinite(confidence).all():
            raise ValueError("Nonfinite observed inputs or invalid mask")
        video_valid, radar_valid = (confidence >= .3).any(1), mask.any(1)
        if self.method in ("rgb_single", "rgb_matched", "radar_single"):
            if self.method == "radar_single":
                _, features = self.point_encoder(points, mask)
                radar = self.radar_head(features).reshape(-1, 17, 3)
                video, video_valid, fused = torch.zeros_like(radar), torch.zeros_like(video_valid), radar
            else:
                features = self.rgb_encoder(rgb) * video_valid[:, None]
                video = self.video_head(features).reshape(-1, 17, 3)
                radar, radar_valid, fused = torch.zeros_like(video), torch.zeros_like(radar_valid), video
            return {"video": video, "radar": radar, "fused": fused, "video_valid": video_valid,
                    "radar_valid": radar_valid, "both": video_valid & radar_valid}
        video_features = self.rgb_encoder(rgb) * video_valid[:, None]
        point_features, radar_features = self.point_encoder(points, mask)
        video = self.video_head(video_features).reshape(-1, 17, 3)
        radar = self.radar_head(radar_features).reshape(-1, 17, 3)
        count = mask.sum(1).to(rgb.dtype) / 196
        result = {"video": video, "radar": radar, "video_valid": video_valid,
                  "radar_valid": radar_valid, "both": video_valid & radar_valid}

        def evidence(base, candidate):
            # Gate evidence must not let the auxiliary BCE manipulate pose values.
            # Apply identically to supervised and unsupervised gate ablations.
            base, candidate = base.detach(), candidate.detach()
            bone = bone_disagreement(base + self.target_template_m, candidate + self.target_template_m)
            if self.method == "utility_nobone":
                bone = bone * 0
            return torch.cat((base, candidate, (base - candidate).abs(), confidence[..., None],
                              count[:, None, None].expand(-1, 17, 1), bone[..., None]), -1)

        if self.method == "concat":
            fused = self.concat_head(torch.cat((video_features, radar_features), -1)).reshape(-1, 17, 3)
        elif self.method == "late_average":
            fused = (video + radar) / 2
        elif self.method == "quality_gate":
            logits = self.gate(evidence(video, radar)).squeeze(-1)
            weight = logits.sigmoid()
            fused = video + weight[..., None] * (radar - video)
            result.update(gate=weight, gate_logits=logits)
        elif self.method == "precision":
            # Isotropic per-joint conditional variance, m², no joint posterior claim.
            vv = F.softplus(self.video_variance(video_features)) + 1e-5
            rv = F.softplus(self.radar_variance(radar_features)) + 1e-5
            weight = vv / (vv + rv)
            fused = video + weight[..., None] * (radar - video)
            result.update(video_variance=vv, radar_variance=rv, gate=weight)
        else:
            projected = self.video_projection(video_features)[:, None].expand(-1, 17, -1)
            joint = self.joint_embedding[None].expand(len(rgb), -1, -1)
            query = self.query(torch.cat((projected, joint), -1))
            scores = torch.einsum("bjh,bph->bjp", query, self.key(point_features)) / 8
            scores = scores.masked_fill(~mask[:, None], -torch.inf)
            scores = torch.where(radar_valid[:, None, None], scores, torch.zeros_like(scores))
            attention = scores.softmax(-1).masked_fill(~mask[:, None], 0.)
            context = attention @ self.value(point_features)
            residual = self.residual(torch.cat((projected, context, joint), -1))
            # Bounded candidate correction, same in every attention/gate ablation.
            candidate = video + .3 * torch.tanh(residual)
            result["candidate"] = candidate
            if self.method == "joint_attention":
                fused = candidate
            else:
                features = evidence(video, candidate)
                if self.method == "utility_global":
                    features = features.mean(1, keepdim=True).expand(-1, 17, -1)
                logits = self.gate(features).squeeze(-1)
                weight = logits.sigmoid()
                fused = video + weight[..., None] * (candidate - video)
                result.update(gate=weight, gate_logits=logits)
        # Exact learned single-branch fallbacks when a whole modality is absent.
        fused = torch.where(~radar_valid[:, None, None], video, fused)
        fused = torch.where(~video_valid[:, None, None], radar, fused)
        result["fused"] = fused
        return result


def weighted_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def fusion_loss(output, target, method, random_generator=None):
    """Shared pose/branch/candidate objectives; risk coefficient fixed at .001."""
    target = target.reshape(-1, 17, 3)
    pose = (output["fused"] - target).square().mean()
    video = weighted_mean((output["video"] - target).square().mean((1, 2)), output["video_valid"])
    radar = weighted_mean((output["radar"] - target).square().mean((1, 2)), output["radar_valid"])
    total = pose + .25 * (video + radar)
    components = {"pose_MSE_m2": pose.detach(), "video_MSE_m2": video.detach(), "radar_MSE_m2": radar.detach()}
    if "candidate" in output:
        candidate_loss = weighted_mean((output["candidate"] - target).square().mean((1, 2)), output["both"])
        total = total + .25 * candidate_loss
        components["candidate_MSE_m2"] = candidate_loss.detach()
    if method.startswith("utility"):
        labels = utility_targets(output["video"], output["candidate"], target)
        valid = output["both"][:, None].expand_as(labels)
        if method == "utility_random":
            selected = labels[valid]
            order = torch.randperm(selected.numel(), generator=random_generator, device=labels.device)
            labels = labels.clone()
            labels[valid] = selected[order]
        risk = weighted_mean(F.binary_cross_entropy_with_logits(output["gate_logits"], labels, reduction="none"), valid)
        total = total + .001 * risk
        components["utility_BCE"] = risk.detach()
    if method == "precision":
        terms = []
        for name in ("video", "radar"):
            # Detaching squared mean error makes this an auxiliary variance fit.
            error = (output[name] - target).square().mean(-1).detach()
            variance = output[name + "_variance"]
            nll = .5 * (error / variance + variance.log())
            terms.append(weighted_mean(nll, output[name + "_valid"][:, None].expand_as(nll)))
        variance_loss = sum(terms)
        total = total + .001 * variance_loss
        components["variance_NLL"] = variance_loss.detach()
    return total, components
