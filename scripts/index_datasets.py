"""为本地公开数据集生成可复用的 CSV 清单。"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_gmdcsa24() -> int:
    root = ROOT / "data" / "raw" / "GMDCSA24"
    rows: list[dict[str, object]] = []
    for subject_dir in sorted(root.glob("Subject *")):
        match = re.search(r"(\d+)$", subject_dir.name)
        if not match:
            continue
        subject = int(match.group(1))
        split = "val" if subject == 4 else "train"
        for class_dir, label in (("ADL", "adl"), ("Fall", "fall")):
            metadata: dict[str, dict[str, str]] = {}
            metadata_path = subject_dir / f"{class_dir}.csv"
            if metadata_path.is_file():
                with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
                    for item in csv.DictReader(handle):
                        name = (item.get("File Name") or "").strip()
                        if name:
                            metadata[name] = item
            for video in sorted((subject_dir / class_dir).glob("*.mp4")):
                item = metadata.get(video.name, {})
                rows.append(
                    {
                        "path": video.relative_to(ROOT).as_posix(),
                        "split": split,
                        "label": label,
                        "subject": subject,
                        "duration_s": (item.get("Length (seconds)") or "").strip(),
                        "description": (item.get("Description") or "").strip(),
                        "classes": (item.get(" Classes") or item.get("Classes") or "").strip(),
                    }
                )
    _write(
        MANIFEST_DIR / "gmdcsa24.csv",
        ["path", "split", "label", "subject", "duration_s", "description", "classes"],
        rows,
    )
    return len(rows)


def index_upfall_skeletons() -> int:
    root = ROOT / "data" / "raw" / "UP-Fall-3D-skeletons" / "extracted"
    rows: list[dict[str, object]] = []
    for sequence in sorted(root.rglob("*.csv")):
        labels: list[str] = []
        with sequence.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            label_key = next((key for key in (reader.fieldnames or []) if key.upper() == "LABEL"), None)
            if label_key:
                labels = [row.get(label_key, "").strip() for row in reader]
        positive = sum(value == "1" for value in labels)
        rows.append(
            {
                "path": sequence.relative_to(ROOT).as_posix(),
                "subject": sequence.parts[-2],
                "sequence": sequence.stem,
                "frames": len(labels),
                "impact_frames": positive,
                "non_impact_frames": len(labels) - positive,
            }
        )
    _write(
        MANIFEST_DIR / "upfall_3d_skeletons.csv",
        ["path", "subject", "sequence", "frames", "impact_frames", "non_impact_frames"],
        rows,
    )
    return len(rows)


def main() -> None:
    gmdcsa_count = index_gmdcsa24()
    upfall_count = index_upfall_skeletons()
    print(f"indexed GMDCSA24 videos: {gmdcsa_count}")
    print(f"indexed UP-Fall skeleton sequences: {upfall_count}")


if __name__ == "__main__":
    main()
