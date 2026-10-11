"""跌倒指标、短视频解码及真实训练中断恢复的检查。"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts import train_video
from scripts.predict_video import predict
from yanglao_monitor.video import _read_frames


def test_collapsed_predictions_and_invalid_metrics():
    metrics = train_video.classification_metrics([1, 1, 1, 1], [0, 0, 1, 1], 1)
    assert metrics["confusion_matrix"] == [[0, 2], [0, 2]]
    assert metrics["fall_recall"] == 1 and metrics["adl_recall"] == 0
    assert math.isclose(metrics["val_macro_f1"], 1 / 3)
    assert train_video.classification_metrics([0, 1], [0, 1], 0)["val_macro_f1"] == 1
    with pytest.raises(ValueError):
        train_video.classification_metrics([0], [0, 1], 1)
    with pytest.raises(ValueError):
        train_video.classification_metrics([], [], 1)


def test_short_video_repeats_frames_without_swapping_rgb(tmp_path):
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.full((16, 16, 3), [0, 0, 240], dtype=np.uint8))
    writer.release()
    clip = _read_frames(path, 8, (16, 16))
    assert clip.shape == (3, 8, 16, 16)
    assert (clip[0].mean(dim=(1, 2)) > 0.8).all()
    assert clip[2].max() < 0.1
    with pytest.raises(RuntimeError):
        _read_frames(tmp_path / "missing.mp4", 8, (16, 16))
    model = train_video.TinyVideoNet(2)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([1.0, -1.0]))
    checkpoint = tmp_path / "inference.pt"
    torch.save({"model": model.state_dict(), "classes": ["adl", "fall"], "clip_len": 2,
                "read_size": (16, 16), "model_name": "tiny"}, checkpoint)
    result = predict(str(path), str(checkpoint))
    assert result["predicted"] == "adl" and result["scope"] == "whole_video"
    assert math.isclose(result["fall_probability"], 1 / (1 + math.exp(2)), rel_tol=1e-5)


def test_augmentation_preserves_temporal_consistency():
    x = torch.linspace(0, 0.5, 8).reshape(1, 1, 1, 1, 8).expand(2, 3, 4, 8, 8)
    y = train_video.prepare_batch(x, augment=True)
    assert torch.equal(y[:, :, 0], y[:, :, 3])
    assert y.min() >= 0 and y.max() <= 1


def test_live_preprocessing_matches_training_on_identical_frames(tmp_path):
    from app.video_model import TrainedVideoModel
    from torchvision.models.video import R3D_18_Weights

    path = tmp_path / "texture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for _ in range(16):
        writer.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    model = TrainedVideoModel.__new__(TrainedVideoModel)
    model.torch, model.device, model.read_size, model.clip_len = torch, "cpu", (171, 128), 16
    model.preprocess = R3D_18_Weights.DEFAULT.transforms()
    offline = _read_frames(path, 16, (171, 128)).unsqueeze(0)
    expected = train_video.prepare_batch(offline, model.preprocess)
    assert torch.allclose(model.prepare_clip(frames), expected, atol=1e-6)


def test_preprocessing_receives_contiguous_temporal_batches():
    def preprocess(x):
        # torchvision 0.20 使用 view，多个片段时必须连续。
        assert x.is_contiguous()
        assert x.shape == (2, 4, 3, 8, 8)
        return x.view(-1, 3, 8, 8)
    y = train_video.prepare_batch(torch.zeros(2, 3, 4, 8, 8, dtype=torch.uint8), preprocess)
    assert y.shape == (8, 3, 8, 8)


def test_low_light_training_keeps_valid_input_range_and_normal_clips():
    clips = torch.full((16,3,8,32,32), .7)
    torch.manual_seed(0)
    result = train_video.prepare_batch(clips, augment=True, low_light=True)
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    means = result.mean((1,2,3,4))
    assert (means < .4).any() and (means > .55).any(), 'train on dark and normal conditions together'
    assert torch.equal(clips, torch.full_like(clips,.7)), 'augmentation must not corrupt cached clips'


def test_partial_body_views_remove_different_evidence_without_artificial_motion():
    source = torch.linspace(0,1,20).reshape(1,1,1,20,1).expand(2,3,8,20,20).clone()
    original = source.clone()
    views = {case:train_video.partial_view(source,case) for case in train_video.PARTIAL_CASES}
    assert views['upper_body'].mean() < source.mean() < views['lower_body'].mean()
    assert not torch.equal(views['center_occlusion'],source)
    for view in views.values():
        assert view.shape == source.shape and torch.equal(view[:,:,0],view[:,:,-1])
        assert torch.isfinite(view).all() and 0 <= view.min() <= view.max() <= 1
    assert torch.equal(source,original)
    with pytest.raises(ValueError):train_video.partial_view(source,'missing')
    with pytest.raises(ValueError):train_video.partial_view(source,'upper_body',.1)
    torch.manual_seed(0)
    augmented = train_video.prepare_batch(source,augment=True,partial=True)
    assert torch.equal(augmented[:,:,0],augmented[:,:,-1])
    assert torch.equal(source,original)


@pytest.mark.parametrize("architecture", ["tiny", "pose_tcn"])
def test_interrupted_training_resumes_exactly(tmp_path, monkeypatch, architecture):
    class Dataset:
        def __init__(self, root, *args, **kwargs):
            self.root = Path(root)
            self.classes = ["adl", "fall"]
            self.class_to_idx = {"adl": 0, "fall": 1}
            self.samples = [(self.root / f"{i}.mp4", i % 2) for i in range(4)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            shape = (3, 2, 33) if architecture == "pose_tcn" else (3, 2, 8, 8)
            return torch.full(shape, index / 4), index % 2

    monkeypatch.setattr(train_video, "VideoFolderDataset", Dataset)
    args = argparse.Namespace(data_root=str(tmp_path), manifest=None, model=architecture, clip_len=2, size=8,
                              workers=0, batch_size=2, cache_clips=True, cpu=True, amp=False, lr=1e-3,
                              weight_decay=1e-4, epochs=2, resume=None, output=str(tmp_path / "reference.pt"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        train_video.train(args)
        reference = torch.load(tmp_path / "reference_last.pt", weights_only=False)
        evaluate = train_video.evaluate
        calls = 0

        def interrupt(*values):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return evaluate(*values)

        args.output = str(tmp_path / "resumed.pt")
        monkeypatch.setattr(train_video, "evaluate", interrupt)
        torch.manual_seed(0)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_video.train(args)
        monkeypatch.setattr(train_video, "evaluate", evaluate)
        args.resume = str(tmp_path / "resumed_last.pt")
        # New CLI defaults must still resume checkpoints written before these options existed.
        args.teacher_checkpoint,args.teacher_weight,args.risk_loss_weight = None,0.,0.
        train_video.train(args)
        resumed = torch.load(args.resume, weights_only=False)
        assert resumed["epoch"] == 2
        assert len(json.loads((tmp_path / "resumed.json").read_text())["history"]) == 2
        for name, value in reference["model"].items():
            assert torch.equal(value, resumed["model"][name]), name
        best = torch.load(args.output, weights_only=False)
        report = json.loads((tmp_path / "resumed_evaluation.json").read_text())
        assert report["selected_epoch"] == best["epoch"]
        assert report["metrics"]["val_macro_f1"] == best["metrics"]["val_macro_f1"]
    finally:
        torch.set_num_threads(previous_threads)
