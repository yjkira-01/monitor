"""训练 TinyVideoNet 或预训练 R3D-18；保存指标、断点和逐视频预测。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yanglao_monitor.video import ACTIVITY_CLASSES, TinyVideoNet, VideoFolderDataset, VideoManifestDataset  # noqa: E402
from yanglao_monitor.pose_temporal import PoseManifestDataset, PoseTemporalNet, pose_features  # noqa: E402
from yanglao_monitor.checkpoint_io import save_training_checkpoint  # noqa: E402


def binary_metrics(predictions: list[int], labels: list[int], positive_index: int) -> dict[str, float]:
    if len(predictions) != len(labels) or not labels:
        raise ValueError("预测和标签必须非空且等长")
    tp = sum(p == positive_index and y == positive_index for p, y in zip(predictions, labels))
    fp = sum(p == positive_index and y != positive_index for p, y in zip(predictions, labels))
    fn = sum(p != positive_index and y == positive_index for p, y in zip(predictions, labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-8)}


def classification_metrics(predictions: list[int], labels: list[int], fall_index: int) -> dict:
    if any(value not in (0, 1) for value in predictions + labels) or fall_index not in (0, 1):
        raise ValueError("仅支持两类跌倒检测")
    fall = binary_metrics(predictions, labels, fall_index)
    adl = binary_metrics(predictions, labels, 1 - fall_index)
    confusion = [[0, 0], [0, 0]]
    for pred, label in zip(predictions, labels):
        confusion[label][pred] += 1
    return {
        "val_acc": sum(p == y for p, y in zip(predictions, labels)) / len(labels),
        "val_macro_f1": (fall["f1"] + adl["f1"]) / 2,
        "val_f1": fall["f1"], "fall_recall": fall["recall"],
        "fall_precision": fall["precision"], "adl_recall": adl["recall"],
        "confusion_matrix": confusion,
    }


def activity_metrics(probabilities, labels, classes):
    if not labels or len(probabilities) != len(labels) or set(classes) != set(ACTIVITY_CLASSES):
        raise ValueError('Multistate evaluation requires the ten annotated activities')
    if any(len(p) != len(classes) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in probabilities) or any(y not in range(len(classes)) for y in labels):
        raise ValueError('Invalid activity probabilities or labels')
    predictions = [max(range(len(p)),key=p.__getitem__) for p in probabilities]
    confusion = [[0]*len(classes) for _ in classes]
    for y,p in zip(labels,predictions):confusion[y][p] += 1
    per_class = {}
    for i,name in enumerate(classes):
        tp = confusion[i][i]; support = sum(confusion[i]); predicted = sum(row[i] for row in confusion)
        precision,recall = tp/max(predicted,1),tp/max(support,1)
        per_class[name] = {'precision':precision,'recall':recall,'f1':2*tp/max(support+predicted,1),'support':support}
    risk = [classes.index(name) for name in ('fall','fallen')]
    risk_metrics = classification_metrics([int(sum(p[i] for i in risk) >= .5) for p in probabilities],[int(y in risk) for y in labels],1)
    return {**risk_metrics,'val_acc':sum(p==y for p,y in zip(predictions,labels))/len(labels),
            'val_macro_f1':sum(v['f1'] for v in per_class.values())/len(classes),'confusion_matrix':confusion,
            'classes':classes,'per_class':per_class,'risk_confusion_matrix':risk_metrics['confusion_matrix'],
            'risk_semantics':'fall or fallen; lying is a separate normal posture, not sleep'}


def domain_metrics(probabilities, labels, rows, classes=None):
    from collections import defaultdict
    if len(rows) != len(labels) or len(probabilities) != len(labels):
        raise ValueError('Validation rows and predictions are not aligned')
    groups = defaultdict(list)
    for i,row in enumerate(rows):groups[row['dataset']].append(i)
    result = {}
    for name,indices in groups.items():
        scores,targets = [probabilities[i] for i in indices],[labels[i] for i in indices]
        metrics = activity_metrics(scores,targets,classes) if classes else classification_metrics([int(p>=.5) for p in scores],targets,1)
        if classes:
            supported = [v['f1'] for v in metrics['per_class'].values() if v['support']]
            metrics['supported_macro_f1'] = sum(supported)/len(supported)
        else:metrics['supported_macro_f1'] = metrics['val_macro_f1']
        result[name] = {**metrics,'windows':len(indices),'videos':len({rows[i]['path'] for i in indices})}
    return result


def sampling_weights(rows, tempered=False):
    from collections import Counter
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Sampling requires training rows only')
    groups = [(r.get('dataset','unknown'),r['label']) for r in rows]
    counts,domains = Counter(groups),Counter(g[0] for g in groups)
    if not tempered:return [1/counts[g] for g in groups]
    # Equal domain mass; square-root class correction caps a clip at 5x its domain's uniform probability.
    raw = [min(math.sqrt(domains[g[0]]/counts[g]),5.) for g in groups]
    totals = Counter()
    for group,weight in zip(groups,raw):totals[group[0]] += weight
    return [weight/totals[group[0]] for group,weight in zip(groups,raw)]


def bounded_sampling_weights(rows):
    from collections import Counter
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Sampling requires training rows only')
    counts = Counter((r['dataset'], r['label']) for r in rows)
    raw = [1/math.sqrt(counts[r['dataset'], r['label']]) for r in rows]
    # Small domains retain coverage without repeating any window more than 5x in expectation.
    ceiling = min(raw)*5
    raw = [min(value, ceiling) for value in raw]
    total = sum(raw)
    return [value/total for value in raw]


def accumulation_weight(index, samples, batch_samples, batch_size, accumulation):
    group_start = index//accumulation*accumulation*batch_size
    return batch_samples/min(accumulation*batch_size, samples-group_start)


def validate_execution_resume(previous, args):
    performance = getattr(args, 'performance_resume', False)
    if performance:
        if args.model != 'internvideo2_base':
            raise ValueError('Performance resume is only supported for InternVideo2')
        if previous['batch_size'] * previous.get('accumulate', 1) != args.batch_size * args.accumulate:
            raise ValueError('Performance resume must preserve the effective batch')
    for key, default in (('batch_size', None), ('accumulate', 1), ('eval_batch_size', 0),
                         ('checkpoint_blocks', 12), ('fused_optimizer', False)):
        if previous.get(key, default) != getattr(args, key, default) and not performance:
            raise ValueError(f'恢复配置与断点不一致: {key}')


def restore_optimizer_execution(optimizer, fused):
    for group in optimizer.param_groups:
        group['fused'] = fused
        for parameter in group['params']:
            state = optimizer.state[parameter]
            if isinstance(state.get('step'), torch.Tensor):
                state['step'] = state['step'].to(parameter.device if fused else 'cpu')


def risk_logits(logits, classes):
    risk = [classes.index(c) for c in ('fall','fallen')]
    normal = [i for i in range(len(classes)) if i not in risk]
    return torch.stack((logits[:,normal].logsumexp(1),logits[:,risk].logsumexp(1)),1)


def activity_loss(logits, labels, classes, risk_weight=0., teacher_logits=None, teacher_weight=0.):
    from torch.nn import functional as F
    loss = F.cross_entropy(logits,labels)
    targets = (labels == classes.index('fall')) | (labels == classes.index('fallen'))
    grouped = risk_logits(logits.float(),classes)
    if risk_weight:loss = loss+ risk_weight*F.cross_entropy(grouped,targets.long())
    if teacher_logits is not None and teacher_weight:
        teacher_logits = teacher_logits.detach().float()
        confidence,predicted = teacher_logits.softmax(1).max(1)
        agreed = (predicted == targets.long()) & (confidence >= .8)
        if agreed.any():
            # Keep old evidence only when the teacher agrees with the author's risk label.
            student = risk_logits(logits.float()/2,classes).log_softmax(1)
            loss = loss+teacher_weight*4*F.kl_div(student[agreed],(teacher_logits/2).softmax(1)[agreed],reduction='batchmean')
    return loss


def quality_view(clips, scale=.35):
    from torch.nn import functional as F
    if clips.ndim != 5 or not .1 <= scale <= 1:
        raise ValueError('Invalid video quality transform')
    batch,channels,length,height,width = clips.shape
    flat = clips.transpose(1,2).reshape(batch*length,channels,height,width)
    small = F.interpolate(flat,size=(max(1,round(height*scale)),max(1,round(width*scale))),mode='bilinear',align_corners=False)
    blurred = F.avg_pool2d(F.pad(small,(1,1,1,1),mode='replicate'),3,stride=1)
    restored = F.interpolate(blurred,size=(height,width),mode='bilinear',align_corners=False)
    return restored.reshape(batch,length,channels,height,width).transpose(1,2)


def selection_rank(metrics, baseline=None):
    score = metrics.get('selection_macro_f1',metrics['val_macro_f1'])
    if 'by_dataset' not in metrics:
        return (score,metrics['fall_recall'],metrics['val_acc'])
    worst = min(m['supported_macro_f1'] for m in metrics['by_dataset'].values())
    metrics['worst_dataset_macro_f1'] = worst
    metrics['domain_selection_score'] = .5*(score+worst)
    eligible = True
    if baseline:
        eligible = metrics['val_macro_f1'] >= baseline['val_macro_f1']-.02 and metrics['fall_recall'] >= baseline['fall_recall']-.03
        for domain,reference in baseline['by_dataset'].items():
            current = metrics['by_dataset'][domain]
            eligible &= current['supported_macro_f1'] >= reference['supported_macro_f1']-.03
        for case,reference in baseline.get('robustness',{}).items():
            eligible &= metrics['robustness'][case]['val_macro_f1'] >= reference['val_macro_f1']-.03
    metrics['regression_gate_passed'] = bool(eligible)
    return (int(eligible),metrics['domain_selection_score'],metrics['fall_recall'],metrics['val_acc'])


PARTIAL_CASES = ('upper_body', 'lower_body', 'side_crop', 'center_occlusion')


def partial_view(clips: torch.Tensor, case: str, retained: float = .65) -> torch.Tensor:
    """Shared crop or occluder across time; cached source clips stay unchanged."""
    from torch.nn import functional as F
    if case not in PARTIAL_CASES or not .5 <= retained <= .95:
        raise ValueError('Unknown partial view or invalid retained fraction')
    batch, channels, length, height, width = clips.shape
    if case == 'center_occlusion':
        result = clips.clone()
        result[..., :, int(width*.375):int(width*.625)] = clips.mean((1,2,3,4),keepdim=True)
        return result
    cropped = clips[..., :round(height*retained), :] if case == 'upper_body' else clips[..., height-round(height*retained):, :] if case == 'lower_body' else clips[..., :round(width*retained)]
    flat = cropped.transpose(1,2).reshape(batch*length,channels,*cropped.shape[-2:])
    return F.interpolate(flat,size=(height,width),mode='bilinear',align_corners=False).reshape(batch,length,channels,height,width).transpose(1,2)


def prepare_batch(clips: torch.Tensor, preprocess=None, augment: bool = False, strong: bool = False, low_light: bool = False, partial: bool = False, quality: bool = False) -> torch.Tensor:
    if clips.ndim == 4:
        return pose_features(clips, augment=augment)
    if clips.dtype == torch.uint8:
        clips = clips.float() / 255
    if augment:
        flip = torch.rand(clips.shape[0], 1, 1, 1, 1, device=clips.device) < 0.5
        clips = torch.where(flip, clips.flip(-1), clips)
        brightness = 0.8 + 0.4 * torch.rand(clips.shape[0], 1, 1, 1, 1, device=clips.device)
        clips = (clips * brightness).clamp(0, 1)
        if strong:
            # Whole-clip transforms preserve timing while varying apparent distance and lighting.
            from torch.nn import functional as F
            batch, channels, length, height, width = clips.shape
            theta = torch.zeros(batch, 2, 3, device=clips.device)
            scale = .85 + .75 * torch.rand(batch, device=clips.device)
            theta[:, 0, 0] = theta[:, 1, 1] = scale
            theta[:, :, 2] = .2 * torch.rand(batch, 2, device=clips.device)-.1
            flat = clips.transpose(1, 2).reshape(batch*length, channels, height, width)
            grid = F.affine_grid(theta.repeat_interleave(length, 0), flat.shape, align_corners=False)
            clips = F.grid_sample(flat, grid, align_corners=False).reshape(batch, length, channels, height, width).transpose(1, 2)
            contrast = .7 + .6 * torch.rand(batch, 1, 1, 1, 1, device=clips.device)
            mean = clips.mean((1, 3, 4), keepdim=True)
            clips = ((clips-mean)*contrast+mean).clamp(0, 1)
        if low_light:
            shape = (clips.shape[0], 1, 1, 1, 1)
            night = torch.rand(shape, device=clips.device) < .6
            exposure = .12+.58*torch.rand(shape, device=clips.device)
            gamma = 1.+1.5*torch.rand(shape, device=clips.device)
            sigma = .004+.018*torch.rand(shape, device=clips.device)
            dark = (clips.pow(gamma)*exposure+torch.randn_like(clips)*sigma).clamp(0, 1)
            # Mix raw and corrected dark clips; inference uses a shared gamma per clip.
            brightness = dark.mean((1, 2, 3, 4), keepdim=True).clamp(min=1/255, max=.99)
            correction = (math.log(80/255)/brightness.log()).clamp(.3, 1.)
            corrected = dark.pow(correction)
            dark = torch.where(torch.rand(shape, device=clips.device) < .5, corrected, dark)
            clips = torch.where(night, dark, clips)
        if partial:
            for index in range(clips.shape[0]):
                choice = int(torch.randint(0, 8, ()).item())
                if choice < len(PARTIAL_CASES):
                    retained = .6+.3*float(torch.rand(()))
                    clips[index:index+1] = partial_view(clips[index:index+1], PARTIAL_CASES[choice], retained)
        if quality:
            degraded = quality_view(clips,.25+.4*float(torch.rand(())))
            clips = torch.where(torch.rand((clips.shape[0],1,1,1,1),device=clips.device)<.5,degraded,clips)
    return preprocess(clips.permute(0, 2, 1, 3, 4).contiguous()) if preprocess else clips


@torch.no_grad()
def evaluate(model, loader, device, preprocess, fall_index: int, dark=False, partial=None, classes=None, quality=False) -> tuple[dict, list, list[int]]:
    model.eval()
    probabilities, labels = [], []
    generator = torch.Generator(device=device).manual_seed(20261008)
    for clips, targets in loader:
        clips = clips.to(device, non_blocking=True)
        if quality:
            clips = quality_view(clips.float()/255 if clips.dtype == torch.uint8 else clips)
        if partial:
            clips = clips.float()/255 if clips.dtype == torch.uint8 else clips
            clips = partial_view(clips, partial)
        if dark:
            clips = clips.float()/255 if clips.dtype == torch.uint8 else clips
            noise = torch.randn(clips.shape, generator=generator, device=device)*.006
            clips = (clips.pow(2.2)*.25+noise).clamp(0,1)
            brightness = clips.mean((1,2,3,4),keepdim=True).clamp(min=1/255,max=.99)
            clips = clips.pow((math.log(80/255)/brightness.log()).clamp(.3,1.))
        clips = prepare_batch(clips, preprocess)
        amp = device.type == 'cuda' and getattr(model, 'amp_evaluation', False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            scores = model(clips).float().softmax(1)
        probabilities.extend((scores if classes else scores[:,fall_index]).cpu().tolist())
        labels.extend(targets.tolist())
    if classes:
        return activity_metrics(probabilities,labels,classes), probabilities,labels
    predictions = [fall_index if p >= 0.5 else 1 - fall_index for p in probabilities]
    return classification_metrics(predictions, labels, fall_index), probabilities, labels


def cache_dataset(dataset, workers: int) -> TensorDataset:
    clips, labels = [], []
    for index, (x, y) in enumerate(DataLoader(dataset, batch_size=1, num_workers=workers), 1):
        # Keep private tensors: retaining worker-shared labels holds one file descriptor per window.
        clips.append((x * 255).round().to(torch.uint8) if x.ndim == 5 else x.float().clone())
        labels.append(y.clone())
        if index % 200 == 0:
            print(f"caching {index}/{len(dataset)}", flush=True)
    result = TensorDataset(torch.cat(clips), torch.cat(labels))
    print(f"cached {len(result)} clips, {result.tensors[0].numel()*result.tensors[0].element_size() / 1024**2:.1f} MiB", flush=True)
    return result


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    read_size = (171, 128) if args.model == "r3d18" else (args.size, args.size)
    manifest_sha256 = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest() if args.manifest else None
    if args.manifest:
        with Path(args.manifest).open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        validate_splits(rows)
        dataset = PoseManifestDataset if args.model == "pose_tcn" else VideoManifestDataset
        train_set = dataset(args.manifest, "train", args.clip_len, read_size)
        val_set = dataset(args.manifest, "val", args.clip_len, read_size, classes=train_set.classes)
        if args.model == 'internvideo2_base':
            train_set.preserve_aspect = val_set.preserve_aspect = True
    else:
        train_set = VideoFolderDataset(Path(args.data_root) / "train", args.clip_len, read_size)
        val_set = VideoFolderDataset(Path(args.data_root) / "val", args.clip_len, read_size, classes=train_set.classes)
    multi_state = getattr(args,'multi_state',False)
    if multi_state and (train_set.classes != sorted(ACTIVITY_CLASSES) or {y for _,y in val_set.samples} != set(range(10))):
        raise ValueError('十类动作在训练/验证中必须全部有样本')
    if not multi_state and (len(train_set.classes) != 2 or "fall" not in train_set.class_to_idx):
        raise ValueError("训练集必须有两类，其中跌倒标签为 fall")
    if {p.resolve() for p, _ in train_set.samples} & {p.resolve() for p, _ in val_set.samples}:
        raise ValueError("同一视频不能同时进入训练和验证")
    fall_index = train_set.class_to_idx["fall"]
    val_paths = [str(p.relative_to(val_set.root)) for p, _ in val_set.samples]
    if getattr(args, 'cache_dir', None):
        from yanglao_monitor.clip_cache import disk_cached_dataset
        signature = {'schema': 1, 'manifest_sha256': manifest_sha256, 'clip_len': args.clip_len,
                     'read_size': read_size, 'classes': train_set.classes,
                     'preserve_aspect': getattr(train_set, 'preserve_aspect', False)}
        train_data = disk_cached_dataset(train_set, args.cache_dir, {**signature, 'split': 'train'}, args.workers)
        val_data = disk_cached_dataset(val_set, args.cache_dir, {**signature, 'split': 'val'}, args.workers)
    else:
        train_data = cache_dataset(train_set, args.workers) if args.cache_clips else train_set
        val_data = cache_dataset(val_set, args.workers) if args.cache_clips else val_set
    workers = 0 if args.cache_clips else args.workers
    options = {"num_workers": workers, "pin_memory": device.type == "cuda", "persistent_workers": workers > 0}
    sampler = None
    if getattr(args, "balanced_sampling", False) or getattr(args,'domain_balanced_sampling',False) or getattr(args,'bounded_sampling',False):
        if not args.manifest:
            raise ValueError("Balanced sampling requires a manifest")
        sampling_rows = [r for r in rows if r['split'] == 'train']
        weights = bounded_sampling_weights(sampling_rows) if getattr(args,'bounded_sampling',False) else sampling_weights(sampling_rows,getattr(args,'domain_balanced_sampling',False))
        sampler = WeightedRandomSampler(weights,len(sampling_rows),replacement=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, **options)
    val_loader = DataLoader(val_data, batch_size=getattr(args, 'eval_batch_size', 0) or args.batch_size, **options)

    preprocess = None
    if args.model == "r3d18":
        from torchvision.models.video import R3D_18_Weights, r3d_18
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=None if args.resume or getattr(args, 'init_checkpoint', None) else weights)
        model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
        train_layers = ("layer3.", "layer4.", "fc.") if getattr(args, "unfreeze_layer3", False) else ("layer4.", "fc.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(train_layers))
        preprocess = weights.transforms()
        params = [{"params": model.layer4.parameters(), "lr": args.lr},
                  {"params": model.fc.parameters(), "lr": args.lr * 10}]
        if getattr(args, "unfreeze_layer3", False):
            params.append({"params": model.layer3.parameters(), "lr": args.lr * .3})
    elif args.model == 'internvideo2_base':
        from yanglao_monitor.foundation_video import build_internvideo2, normalize_video, layer_parameters
        model = build_internvideo2(len(train_set.classes), args.pretrained_weights,
                                  checkpoint_blocks=getattr(args, 'checkpoint_blocks', 12))
        preprocess = normalize_video
        params = layer_parameters(model, args.lr)
    elif args.model == "pose_tcn":
        model = PoseTemporalNet()
        params = model.parameters()
    else:
        model = TinyVideoNet(num_classes=len(train_set.classes))
        params = model.parameters()
    initial_sha256 = None
    if getattr(args, 'init_checkpoint', None):
        if args.resume:
            raise ValueError('init_checkpoint and resume are mutually exclusive')
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        initial_sha256 = hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest()
        if initial['model_name'] != args.model or initial['clip_len'] != args.clip_len:
            raise ValueError('初始化权重的模型、类别或片长不匹配')
        if multi_state and initial['classes'] == ['adl','fall'] and args.model == 'r3d18':
            missing,unexpected = model.load_state_dict({k:v for k,v in initial['model'].items() if not k.startswith('fc.')},strict=False)
            if set(missing) != {'fc.weight','fc.bias'} or unexpected:raise ValueError('初始化主干不兼容')
        elif initial['classes'] == train_set.classes:
            model.load_state_dict(initial['model'])
        else:raise ValueError('初始化类别不兼容')
    model.to(device)
    teacher,teacher_sha256 = None,None
    if getattr(args,'teacher_checkpoint',None):
        if args.model != 'r3d18' or not multi_state:
            raise ValueError('Risk teacher requires R3D ten-class training')
        teacher_state = torch.load(args.teacher_checkpoint,map_location='cpu',weights_only=True)
        if teacher_state['model_name'] != 'r3d18' or teacher_state['classes'] != ['adl','fall'] or teacher_state['clip_len'] != args.clip_len or tuple(teacher_state['read_size']) != read_size or teacher_state.get('preprocessing') != 'kinetics400_v1':
            raise ValueError('Teacher architecture or preprocessing is incompatible')
        teacher_sha256 = hashlib.sha256(Path(args.teacher_checkpoint).read_bytes()).hexdigest()
        teacher = r3d_18(weights=None)
        teacher.fc = nn.Linear(teacher.fc.in_features,2)
        teacher.load_state_dict(teacher_state['model'])
        teacher.requires_grad_(False).to(device).eval()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                                  fused=True if getattr(args, 'fused_optimizer', False) else None)
    warmup = min(5, max(args.epochs - 1, 0))
    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(step - warmup, args.epochs - warmup) / max(args.epochs - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16, init_scale=1024.)
    criterion = nn.CrossEntropyLoss()
    history, best_rank, recall_rank, first_epoch = [], (-1.0, -1.0, -1.0), (-1.0, -1.0, -1.0), 1
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        initial_sha256 = state.get('initial_checkpoint_sha256')
        if state["manifest_sha256"] != manifest_sha256:
            raise ValueError("恢复训练的数据清单已改变")
        for key in ("model", "clip_len", "size", "epochs", "lr", "weight_decay"):
            if state["config"][key] != vars(args)[key]:
                raise ValueError(f"恢复配置与断点不一致: {key}")
        validate_execution_resume(state['config'], args)
        for key in ("balanced_sampling", "unfreeze_layer3", "strong_augment", "low_light_augment", "partial_body_augment", "multi_state", "patience", "domain_balanced_sampling", "domain_selection", "quality_augment", "risk_loss_weight", "teacher_weight", "teacher_checkpoint"):
            default = None if key == 'teacher_checkpoint' else 0 if key in {'patience','risk_loss_weight','teacher_weight'} else False
            if state["config"].get(key,default) != getattr(args,key,default):
                raise ValueError(f"恢复配置与断点不一致: {key}")
        for key, default in (('bounded_sampling',False),('pretrained_weights',None),('reference_metrics',None)):
            if state['config'].get(key,default) != getattr(args,key,default):
                raise ValueError(f'恢复配置与断点不一致: {key}')
        if state.get('teacher_sha256') != teacher_sha256:raise ValueError('Teacher checkpoint changed while resuming')
        if state["classes"] != train_set.classes:
            raise ValueError("恢复训练类别与断点不一致")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if getattr(args, 'performance_resume', False):
            fused = getattr(args, 'fused_optimizer', False)
            restore_optimizer_execution(optimizer, fused)
            print('performance_resume ' + json.dumps({'epoch': state['epoch'],
                  'previous_batch': state['config']['batch_size'], 'batch': args.batch_size,
                  'accumulate': args.accumulate, 'effective_batch': args.batch_size * args.accumulate,
                  'checkpoint_blocks': getattr(args, 'checkpoint_blocks', 12), 'fused': fused}), flush=True)
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_rank, recall_rank = state["history"], tuple(state["best_rank"]), tuple(state["recall_rank"])
        first_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    last_out = out.with_name(f"{out.stem}_last{out.suffix}")
    recall_out = out.with_name(f"{out.stem}_best_recall{out.suffix}")
    if first_epoch == 1 and any(p.exists() for p in (out, last_out, recall_out)):
        raise FileExistsError("输出已存在，请改用新的 output 或指定 resume")
    if args.resume and (not out.exists() or not recall_out.exists()):
        raise FileNotFoundError("恢复训练需要同一 output 对应的最佳模型文件")
    robust_cases = {}
    eval_options = {'classes':train_set.classes} if multi_state else {}
    if getattr(args,'low_light_augment',False):
        robust_cases['night'] = {'dark': True}
    if getattr(args,'partial_body_augment',False):
        robust_cases.update({case: {'partial': case} for case in PARTIAL_CASES})
    if getattr(args,'quality_augment',False):
        robust_cases['low_quality'] = {'quality': True}
        robust_cases['night_low_quality'] = {'quality':True,'dark':True}
    validation_rows = [r for r in rows if r['split'] == 'val'] if args.manifest else []
    domain_selection = getattr(args,'domain_selection',False)
    baseline = None
    baseline_path = out.with_name(out.stem+'_domain_baseline.json')
    if domain_selection:
        if not args.manifest:raise ValueError('Domain selection requires a manifest')
        if first_epoch == 1:
            if getattr(args,'reference_metrics',None):
                reference = json.loads(Path(args.reference_metrics).read_text(encoding='utf-8-sig'))
                if reference['manifest_sha256'] != manifest_sha256:
                    raise ValueError('The reference must use exactly the same validation manifest')
                baseline = reference['metrics']
            else:
                baseline,probabilities,labels = evaluate(model,val_loader,device,preprocess,fall_index,**eval_options)
                baseline['by_dataset'] = domain_metrics(probabilities,labels,validation_rows,train_set.classes if multi_state else None)
                baseline['robustness'] = {case:evaluate(model,val_loader,device,preprocess,fall_index,**options,**eval_options)[0] for case,options in robust_cases.items()}
            baseline_path.write_text(json.dumps(baseline,indent=2),encoding='utf-8')
        else:baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if first_epoch == 1 and robust_cases and getattr(args,'init_checkpoint',None) and not multi_state:
        baseline, _, _ = evaluate(model,val_loader,device,preprocess,fall_index,**eval_options)
        baseline_cases = {case: evaluate(model,val_loader,device,preprocess,fall_index,**options,**eval_options)[0] for case, options in robust_cases.items()}
        out.with_name(out.stem+'_baseline.json').write_text(json.dumps({'day':baseline,**baseline_cases,'initial_checkpoint_sha256':initial_sha256},indent=2),encoding='utf-8')
    print(f"device={device} model={args.model} train={len(train_data)} val={len(val_data)} amp={amp_enabled} dtype={amp_dtype}", flush=True)
    accumulation = getattr(args, 'accumulate', 1)
    for epoch in range(first_epoch, args.epochs + 1):
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        if args.model == "r3d18":
            # 小数据、小 batch 使用预训练统计量，避免跨受试者 BatchNorm 漂移。
            for module in model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        correct, total, loss_sum = 0, 0, 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (clips, labels) in enumerate(train_loader):
            clips = prepare_batch(clips.to(device, non_blocking=True), preprocess, augment=True, strong=getattr(args, "strong_augment", False), low_light=getattr(args, 'low_light_augment', False), partial=getattr(args, 'partial_body_augment', False),quality=getattr(args,'quality_augment',False))
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(clips)
                with torch.no_grad():teacher_logits = teacher(clips) if teacher else None
                loss = activity_loss(logits,labels,train_set.classes,getattr(args,'risk_loss_weight',0.),teacher_logits,getattr(args,'teacher_weight',0.)) if multi_state else criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失不是有限值")
            weight = accumulation_weight(batch_index, len(train_data), labels.numel(), args.batch_size, accumulation)
            scaler.scale(loss*weight).backward()
            if (batch_index+1)%accumulation == 0 or batch_index+1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += loss.item() * labels.numel()
            if args.model == 'internvideo2_base' and (batch_index == 0 or (batch_index+1)%getattr(args, 'log_every', 100) == 0):
                print(f'epoch={epoch:03d} batch={batch_index+1}/{len(train_loader)} loss={loss.item():.5f}', flush=True)
                duration = max(time.monotonic() - started, 1e-6)
                print(f'throughput epoch={epoch:03d} samples={total} elapsed={duration:.3f} samples_per_second={total/duration:.3f}', flush=True)
        metrics, probabilities, labels = evaluate(model, val_loader, device, preprocess, fall_index,**eval_options)
        if domain_selection:metrics['by_dataset'] = domain_metrics(probabilities,labels,validation_rows,train_set.classes if multi_state else None)
        case_metrics = {case: evaluate(model,val_loader,device,preprocess,fall_index,**options,**eval_options)[0] for case,options in robust_cases.items()}
        if case_metrics:
            metrics['robustness'] = case_metrics
            partial_scores = [v['val_macro_f1'] for k,v in case_metrics.items() if k != 'night']
            groups = [metrics['val_macro_f1']]
            if 'night' in case_metrics:
                groups.append(case_metrics['night']['val_macro_f1'])
            if partial_scores:
                groups.append(sum(partial_scores)/len(partial_scores))
            metrics['selection_macro_f1'] = sum(groups)/len(groups)
        rank = selection_rank(metrics,baseline)
        if args.model == 'internvideo2_base':
            eligible = bool(rank[0]) and metrics['fall_recall'] >= .95 and metrics['fall_precision'] >= .90 and metrics['per_class']['lying']['recall'] >= .80
            metrics['regression_gate_passed'] = eligible
            rank = (int(eligible), *rank[1:])
        metrics['selection_rank'] = rank
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"epoch": epoch, "train_acc": correct / total, "train_loss": loss_sum / total, "lr": current_lr,
               "seconds": time.monotonic() - started,
               "peak_cuda_mib": torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else 0., **metrics}
        history.append(row)
        scheduler.step()
        recall = (metrics["fall_recall"], metrics["val_macro_f1"], metrics["val_acc"])
        improve, improve_recall = rank > best_rank, recall > recall_rank
        best_rank, recall_rank = max(best_rank, rank), max(recall_rank, recall)
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "classes": train_set.classes,
            "clip_len": args.clip_len, "size": args.size, "model_name": args.model, "config": vars(args),
            "read_size": read_size, "preprocessing": "internvideo2_rgb_imagenet_letterbox_v1" if args.model == 'internvideo2_base' else "pose_xy_visibility_body_relative_v1" if args.model == "pose_tcn" else "kinetics400_v1" if preprocess else "rgb_0_1",
            "manifest_sha256": manifest_sha256,
            'initial_checkpoint_sha256': initial_sha256,
            'teacher_sha256': teacher_sha256,
            "history": history, "metrics": row, "best_rank": best_rank, "recall_rank": recall_rank,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        checkpoint_started = time.monotonic()
        selected_paths = [path for path, selected in ((out, improve), (recall_out, improve_recall)) if selected]
        save_training_checkpoint(state, last_out, selected_paths, getattr(args, 'checkpoint_dir', None))
        print(f'checkpoint_io epoch={epoch:03d} seconds={time.monotonic()-checkpoint_started:.3f} '
              f'epoch_wall_seconds={time.monotonic()-started:.3f} selected={len(selected_paths)}', flush=True)
        metrics_out = out.with_suffix(".json")
        temporary = metrics_out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"config": vars(args), "classes": train_set.classes, "history": history,
                                         "best_rank": best_rank, "recall_rank": recall_rank}, indent=2), encoding="utf-8")
        temporary.replace(metrics_out)
        print(f"epoch={epoch:03d} train_acc={row['train_acc']:.3f} val_acc={metrics['val_acc']:.3f} "
              f"macro_f1={metrics['val_macro_f1']:.3f} fall_f1={metrics['val_f1']:.3f} "
              f"recall={metrics['fall_recall']:.3f} adl_recall={metrics['adl_recall']:.3f} sec={row['seconds']:.1f}", flush=True)
        if case_metrics:
            print('robustness '+json.dumps({k:round(v['val_macro_f1'],4) for k,v in case_metrics.items()}),flush=True)
        patience = getattr(args, "patience", 0)
        best_f1_epoch = max(range(len(history)),key=lambda i:tuple(history[i].get('selection_rank',(history[i].get('selection_macro_f1',history[i]['val_macro_f1']),history[i]['fall_recall'],history[i]['val_acc']))))+1
        if patience and epoch-best_f1_epoch >= patience:
            print(f"early_stop epoch={epoch} patience={patience}", flush=True)
            break
    best = torch.load(out, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    metrics, probabilities, labels = evaluate(model, val_loader, device, preprocess, fall_index,**eval_options)
    if domain_selection:
        metrics['by_dataset'] = domain_metrics(probabilities,labels,validation_rows,train_set.classes if multi_state else None)
        metrics.update({key:best['metrics'][key] for key in ('domain_selection_score','worst_dataset_macro_f1','regression_gate_passed')})
    if robust_cases:
        metrics['robustness'] = {case:evaluate(model,val_loader,device,preprocess,fall_index,**options,**eval_options)[0] for case,options in robust_cases.items()}
    torch.save({**{key: best[key] for key in ("model", "model_name", "classes", "clip_len", "size", "read_size", "preprocessing", "metrics")}, 'low_light_trained': getattr(args, 'low_light_augment', False), 'partial_body_trained': getattr(args, 'partial_body_augment', False), 'training_manifest_sha256':manifest_sha256,'teacher_sha256':teacher_sha256},
               out.with_name(f"{out.stem}_inference{out.suffix}"))
    report = {"checkpoint": str(out), "selected_epoch": best["epoch"], "classes": train_set.classes,
              "split": "subject-held-out validation (used for model selection; not an independent test set)",
              "metrics": metrics, "predictions": [
                  {"path": p, **({"start_s": val_set.ranges[i][0], "end_s": val_set.ranges[i][1]} if hasattr(val_set, "ranges") else {}),
                   "label": train_set.classes[y], "fall_probability": sum(prob[train_set.class_to_idx[c]] for c in ('fall','fallen')) if multi_state else prob,
                   **({'activity_scores':dict(zip(train_set.classes,prob))} if multi_state else {}),
                   "predicted": train_set.classes[max(range(len(prob)),key=prob.__getitem__) if multi_state else fall_index if prob >= 0.5 else 1 - fall_index]}
                  for i, (p, y, prob) in enumerate(zip(val_paths, labels, probabilities))]}
    out.with_name(f"{out.stem}_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"complete best_epoch={best['epoch']} metrics={metrics}", flush=True)


def smoke_test() -> None:
    torch.manual_seed(0)
    model = TinyVideoNet(num_classes=2)
    logits = model(torch.rand(2, 3, 8, 64, 64))
    assert logits.shape == (2, 2)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    metrics = classification_metrics([1] * 4, [0, 0, 1, 1], 1)
    assert metrics["confusion_matrix"] == [[0, 2], [0, 2]]
    assert metrics["fall_recall"] == 1 and metrics["adl_recall"] == 0
    assert math.isclose(metrics["val_macro_f1"], 1 / 3)
    print(f"video smoke test passed; loss={loss.item():.4f}")


def validate_splits(rows):
    subjects, paths, parents = {}, {}, {}
    for row in rows:
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        for table, key in ((subjects, (row.get("dataset", ""), row.get("subject", ""))),
                           (paths, row["path"]), (parents, (row.get("dataset", ""), row.get("parent_id", row["path"])))):
            if table is subjects and key[1] in {"", "unknown"}:
                continue
            if key in table and table[key] != split:
                raise ValueError("Subject, parent or video leaked across data splits")
            table[key] = split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/video")
    parser.add_argument("--manifest", help="按 scripts/index_datasets.py 生成的清单训练")
    parser.add_argument("--model", choices=["tiny", "r3d18", "pose_tcn", "internvideo2_base"], default="tiny")
    parser.add_argument('--pretrained-weights', help='Official hashed InternVideo2-Base K400 checkpoint')
    parser.add_argument('--reference-metrics', help='Frozen validation reference with matching manifest SHA256')
    parser.add_argument('--accumulate', type=int, default=1, help='Accumulate microbatches before each optimizer update')
    parser.add_argument('--bounded-sampling', action='store_true', help='Tempered dataset/class sampling, global window probability at most 5/N')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--size", type=int, default=160, help="Tiny 输入大小；R3D 使用官方 128x171 / 112 crop")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-clips", action="store_true", help="缓存 uint8 片段；适用于小数据集")
    parser.add_argument('--cache-dir', help='本地磁盘持久 uint8 缓存；需与 cache-clips 同用')
    parser.add_argument('--checkpoint-dir', help='完整断点在本地盘序列化一次，再同步到 output 所在的持久盘')
    parser.add_argument('--eval-batch-size', type=int, default=0, help='独立验证 batch；0沿用训练 batch')
    parser.add_argument('--checkpoint-blocks', type=int, default=12, help='InternVideo2重复计算层数0..12，减少可使用更多显存提速')
    parser.add_argument('--fused-optimizer', action='store_true', help='使用 CUDA fused AdamW')
    parser.add_argument('--performance-resume', action='store_true', help='显式允许性能配置变化，但保持有效 batch 和学习目标')
    parser.add_argument('--log-every', type=int, default=100, help='训练批次进度记录间隔')
    parser.add_argument("--balanced-sampling", action="store_true", help="按数据集/类别平衡采样")
    parser.add_argument('--domain-balanced-sampling',action='store_true',help='数据集等权，类别平方根修正，单片段概率最多为域内均匀采样的5倍')
    parser.add_argument('--domain-selection',action='store_true',help='逐数据集验证与初始化权重退化门槛，兼顾最差域选模')
    parser.add_argument('--risk-loss-weight',type=float,default=0.,help='十类分类共享输出的跌倒/倒地辅助监督权重')
    parser.add_argument('--teacher-checkpoint',help='旧二分类R3D模型，仅在与作者风险标签一致时提供保留监督')
    parser.add_argument('--teacher-weight',type=float,default=0.,help='标签一致的旧模型蒸馏权重')
    parser.add_argument('--quality-augment',action='store_true',help='时序一致的降采样和模糊，验证低质量及暗光低质量')
    parser.add_argument("--unfreeze-layer3", action="store_true", help="R3D增加layer3小学习率微调")
    parser.add_argument("--strong-augment", action="store_true", help="增加整片距离/平移/对比度增强")
    parser.add_argument('--low-light-augment', action='store_true', help='暗光/噪声/暗部恢复/遮挡增强，保留部分正常光照片段')
    parser.add_argument('--partial-body-augment', action='store_true', help='时序一致的上下半身、侧面截断和中部遮挡，逐条件评估')
    parser.add_argument('--multi-state',action='store_true',help='训练作者标注的10种动作，而非ADL/Fall二类')
    parser.add_argument('--init-checkpoint', help='从兼容推理权重开始新的微调实验，不复用旧优化器')
    parser.add_argument("--patience", type=int, default=0, help="macro-F1不改善轮数；0关闭早停")
    parser.add_argument("--resume", help="同一配置、同一输出路径的 last 断点")
    parser.add_argument("--output", default="artifacts/video_best.pt")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.model == "pose_tcn" and not args.manifest:
        parser.error("pose_tcn requires the cached pose manifest")
    if args.model == 'internvideo2_base' and (not args.manifest or not args.multi_state or args.clip_len != 8 or args.size != 224 or not args.pretrained_weights or not args.amp or not args.domain_selection or not args.reference_metrics):
        parser.error('InternVideo2 requires a ten-class manifest, eight 224px frames, verified pretrained weights, AMP and a matching domain reference')
    if args.model != 'internvideo2_base' and args.pretrained_weights:
        parser.error('pretrained-weights is only used by InternVideo2')
    if args.eval_batch_size < 0:
        parser.error('eval-batch-size must be nonnegative')
    if args.log_every < 1:
        parser.error('log-every must be positive')
    if args.cache_dir and (not args.cache_clips or not args.manifest or args.model == 'pose_tcn'):
        parser.error('cache-dir requires cache-clips and an RGB manifest')
    if not 0 <= args.checkpoint_blocks <= 12 or (args.checkpoint_blocks != 12 and args.model != 'internvideo2_base'):
        parser.error('checkpoint-blocks must be 0..12 and requires InternVideo2')
    if args.fused_optimizer and (args.cpu or not torch.cuda.is_available()):
        parser.error('fused-optimizer requires CUDA')
    if args.performance_resume and (not args.resume or args.model != 'internvideo2_base'):
        parser.error('performance-resume requires an InternVideo2 checkpoint')
    if (args.low_light_augment or args.partial_body_augment or args.multi_state or args.quality_augment) and args.model == 'pose_tcn':
        parser.error('light and crop augmentation require RGB models')
    if min(args.epochs, args.batch_size, args.clip_len, args.size, args.accumulate) < 1 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0 or args.patience < 0:
        parser.error("训练尺寸、轮数、batch 和学习率必须为正，workers/weight-decay 不能为负")
    if not math.isfinite(args.risk_loss_weight) or not math.isfinite(args.teacher_weight) or min(args.risk_loss_weight,args.teacher_weight)<0:
        parser.error('Loss weights must be finite and nonnegative')
    if sum((args.balanced_sampling,args.domain_balanced_sampling,args.bounded_sampling)) > 1:
        parser.error('Choose either legacy or tempered domain sampling')
    if (args.risk_loss_weight or args.teacher_weight or args.teacher_checkpoint) and not args.multi_state:
        parser.error('Risk supervision requires ten-class training')
    if bool(args.teacher_checkpoint) != bool(args.teacher_weight):
        parser.error('Teacher checkpoint and positive teacher weight must be supplied together')
    random.seed(0)
    torch.manual_seed(0)
    train(args)


if __name__ == "__main__":
    main()
