"""Source-preserving X-Fi RGB/radar adapter, not an official protocol reproduction.

Official definitions are read from a pinned source checkout. Only imports and
pretrained/all-five-modality constructors are bypassed. Native RGB, radar TD,
projection and fusion definitions are retained, including their numerical details.
"""

import ast
import hashlib
import sys
from pathlib import Path
from time import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps/paper_baselines"))

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from torch import nn
from torch.nn import functional as F

OFFICIAL = ROOT / "artifacts/paper_baseline_sources_20261013/xfi_checkout/MMFi_HPE"
COMMIT = "9ba74cc98beb2d75a8eadbe74b1b151dc3d39614"
SOURCE_FILES = ("X_Fi.py", "backbones/RGB_benchmark/RGB_ResNet.py",
                "backbones/mmwave_benchmark/mmwave_util.py",
                "backbones/mmwave_benchmark/mmwave_point_transformer_TD.py")
FUSION_DEFINITIONS = {"selective_pos_enc", "linear_projector", "MultiHeadAttention", "qkv_Attention",
                      "FeedForward", "kv_projection", "cross_modal_transformer", "fusion_transformer",
                      "fusion_transformer_block", "regression_Head", "X_Fusion"}


def load_official_definitions(path, namespace, names=None):
    """Compile original definition AST nodes without modifying their bodies."""
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and (names is None or node.name in names)]
    if names is not None and {node.name for node in selected} != names:
        raise ValueError("Pinned source is missing expected definitions")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return [{"definition": node.name, "start_line": node.lineno, "end_line": node.end_lineno,
             "AST_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
            for node in selected]


def base_namespace(name):
    return {"__name__": name, "torch": torch, "nn": nn, "F": F, "np": np,
            "time": time, "rearrange": rearrange, "reduce": reduce, "repeat": repeat}


def official_modules():
    rgb, radar, fusion = [base_namespace("xfi_official_" + x) for x in ("rgb", "radar", "fusion")]
    definitions = {}
    definitions[SOURCE_FILES[1]] = load_official_definitions(OFFICIAL / SOURCE_FILES[1], rgb)
    definitions[SOURCE_FILES[2]] = load_official_definitions(OFFICIAL / SOURCE_FILES[2], radar)
    definitions[SOURCE_FILES[3]] = load_official_definitions(OFFICIAL / SOURCE_FILES[3], radar)
    fusion.update({key: radar[key] for key in ("index_points", "farthest_point_sample")})
    definitions[SOURCE_FILES[0]] = load_official_definitions(OFFICIAL / SOURCE_FILES[0], fusion, FUSION_DEFINITIONS)
    return rgb, radar, fusion, definitions


class XFiRGBRadar(nn.Module):
    """Native 512-d/8-head/depth-4 X-Fi blocks, scratch RGB+radar backbones."""

    def __init__(self, training_prior_m=None, pretrained_directory=None, recalibrate_bn=False):
        super().__init__()
        rgb, radar, fusion, self.official_definitions = official_modules()
        rgb_model, radar_model = rgb["RGB_ResNet18"](), radar["mmwave_PointTransformerReg"]()
        self.backbone_initialization = {"mode": "scratch", "weights": {}}
        if pretrained_directory is not None:
            pretrained_directory = Path(pretrained_directory)
            self.backbone_initialization["mode"] = "author_public_MMFi_pretrained"
            for model, name in ((rgb_model, "RGB_Resnet18.pt"), (radar_model, "mmwave_all_random_TD.pt")):
                path = pretrained_directory / name
                state = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
                    raise ValueError("Native tensor state dict required")
                invalid = [key for key, value in state.items() if not torch.isfinite(value).all()]
                # Recalibration can replace observed BN variance statistics only;
                # any nonfinite parameter or other buffer rejects the checkpoint.
                if invalid and (not recalibrate_bn or any(not key.endswith("running_var") for key in invalid)):
                    raise ValueError("Nonfinite official checkpoint fields: " + ", ".join(invalid))
                model.load_state_dict(state, strict=True)
                self.backbone_initialization["weights"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                                 "nonfinite_original_statistics": invalid}
            if recalibrate_bn:
                for model in (rgb_model, radar_model):
                    for layer in model.modules():
                        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                            layer.reset_running_stats()
                self.backbone_initialization["BN_statistics"] = "reset pending training-only recalibration; learned parameters unchanged"
        self.rgb_extractor = nn.Sequential(*list(rgb_model.children())[:-2])
        self.radar_extractor = radar_model.backbone
        self.backbones_frozen = False
        self.linear_projector = fusion["linear_projector"](512, 512)
        # Keep the official five-slot numbering: RGB=0 and mmWave=2. Inactive
        # slots remain allocated; no hidden width/depth/parameter reduction.
        self.X_Fusion_block = fusion["X_Fusion"](5, 512, 2, 512, 32, 8, 64, 4)
        prior = torch.zeros(17, 3) if training_prior_m is None else torch.as_tensor(training_prior_m, dtype=torch.float32)
        if prior.shape != (17, 3) or not torch.isfinite(prior).all():
            raise ValueError("Training-only 17 x 3 prior required")
        self.register_buffer("training_prior_m", prior.clone())

    def freeze_backbones(self):
        self.backbones_frozen = True
        for model in (self.rgb_extractor, self.radar_extractor):
            model.requires_grad_(False)
            model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.backbones_frozen:
            self.rgb_extractor.eval()
            self.radar_extractor.eval()
        return self

    @torch.no_grad()
    def recalibrate_backbones(self, batches):
        """Caller supplies training observations only; no target argument."""
        encoders = (self.rgb_extractor, self.radar_extractor)
        original_momentum = {}
        original_forwards = {}
        for encoder in encoders:
            encoder.eval()
            for layer in encoder.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    original_momentum[layer] = layer.momentum
                    layer.reset_running_stats()
                    layer.momentum = None
                    layer.train()
                    original_forwards[layer] = layer.forward
                    def stable_statistics_forward(value, original=layer.forward):
                        # Native cuDNN running variance overflows for the public
                        # RGB stem's ~1e10 activations, despite finite FP64 variance.
                        # ATen's native FP32 BN matches the FP64 variance check.
                        previous = torch.backends.cudnn.enabled
                        try:
                            torch.backends.cudnn.enabled = False
                            return original(value)
                        finally:
                            torch.backends.cudnn.enabled = previous
                    layer.forward = stable_statistics_forward
        frames = 0
        try:
            for images, points in batches:
                if len(images) < 2:
                    raise ValueError("BN recalibration requires training batches of at least two")
                self.rgb_extractor(images)
                self.radar_extractor(points)
                frames += len(images)
        finally:
            for layer, momentum in original_momentum.items():
                layer.momentum = momentum
                layer.forward = original_forwards[layer]
                layer.eval()
        if not frames or not all(torch.isfinite(value).all() for encoder in encoders for value in encoder.state_dict().values()):
            raise ValueError("Training-only recalibration produced invalid native backbone statistics")
        self.backbone_initialization["BN_statistics"] = "recomputed from supplied training observations; parameters unchanged"
        self.backbone_initialization["BN_calibration_frames"] = frames
        self.backbone_initialization["BN_calibration_backend"] = "Native ATen FP32 during statistics calibration only; original BN forward restored for frozen inference"

    def native_subset(self, images, points, rgb_present=True, radar_present=True):
        if not (rgb_present or radar_present):
            return self.training_prior_m[None].expand(len(images), -1, -1)
        flag = [rgb_present, False, radar_present, False, False]
        features = []
        if rgb_present:
            value = self.rgb_extractor(images)
            features.append(value.view(len(images), 512, -1).permute(0, 2, 1))
        if radar_present:
            features.append(self.radar_extractor(points)[0])
        projected = self.linear_projector(features, None, flag)
        return self.X_Fusion_block(projected, flag)

    def forward(self, images, points, rgb_available=None, radar_available=None):
        if (images.ndim != 4 or images.shape[1:] != (3, 480, 640) or points.ndim != 3
                or points.shape[0] != len(images) or points.shape[-1] != 5 or not 1 <= points.shape[1] <= 196):
            raise ValueError("Native RGB 480x640 BGR and observed padded BxNx5 points required")
        if not torch.isfinite(images).all() or not torch.isfinite(points).all():
            raise ValueError("Finite public observations required")
        rgb_available = torch.ones(len(images), dtype=torch.bool, device=images.device) if rgb_available is None else rgb_available
        radar_available = points.ne(0).any((1, 2)) if radar_available is None else radar_available
        if (rgb_available.dtype != torch.bool or radar_available.dtype != torch.bool
                or rgb_available.shape != (len(images),) or radar_available.shape != (len(images),)):
            raise ValueError("Aligned boolean modality availability required")
        output = self.training_prior_m[None].expand(len(images), -1, -1).clone()
        for image_flag, point_flag in ((True, True), (True, False), (False, True)):
            selected = (rgb_available == image_flag) & (radar_available == point_flag)
            if selected.any():
                output[selected] = self.native_subset(images[selected], points[selected], image_flag, point_flag)
        return output


def source_manifest():
    return {str(path): hashlib.sha256((OFFICIAL / path).read_bytes()).hexdigest() for path in SOURCE_FILES}
