import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.index_video_windows import fall_intervals, window_label
from yanglao_monitor.video import VideoManifestDataset, _read_frames


def test_coarse_fall_labels_reject_missing_boundaries():
    assert fall_intervals("Fall (FW)[2.6 to 6]; Standing[1 to 2]") == [(2.6, 6.)]
    assert fall_intervals("Falling (SW) [ 2.8 to 6]") == [(2.8, 6.)]
    for text in ("Falling (BW)[2.3 6]", "Falling (FW); Sitting[0 to 1.7]", "Fall[4 to 2]"):
        with pytest.raises(ValueError):
            fall_intervals(text)


def test_transition_and_post_event_gaps_are_not_fabricated_as_normal():
    intervals = [(2., 4.)]
    assert window_label(0, 1.5, intervals) == "adl"
    assert window_label(0, 2.1, intervals) is None
    assert window_label(0, 3., intervals) == "fall"
    assert window_label(4.1, 6., intervals) is None
    assert window_label(0, 5., []) == "adl"


def test_training_reads_selected_window_instead_of_entire_video(tmp_path):
    data = tmp_path / "data"
    manifests = data / "manifests"
    manifests.mkdir(parents=True)
    path = data / "sequence.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for index in range(30):
        color = (0, 0, 220) if index < 10 else (220, 0, 0)
        writer.write(np.full((32, 32, 3), color, dtype=np.uint8))
    writer.release()
    manifest = manifests / "windows.csv"
    fields = ["path", "split", "label", "subject", "start_s", "end_s"]
    rows = [{"path":"data/sequence.avi","split":"train","label":"fall","subject":"1","start_s":"1.1","end_s":"2.8"}]
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)
    dataset = VideoManifestDataset(manifest, "train", 16, (32,32))
    clip, label = dataset[0]
    assert clip[2].mean() > .8 and clip[0].max() < .1
    assert dataset.ranges == [(1.1, 2.8)]
    for start, end in ((2.,1.), (None,1.), (4.,5.), (float("nan"),5.)):
        with pytest.raises(ValueError):
            _read_frames(path, 16, (32,32), start, end)
