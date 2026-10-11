"""官方人物检测/姿态权重；完整 SHA256 验证后原子替换。"""
import hashlib
from pathlib import Path

import requests

MODELS = [
    ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
     "pose_landmarker_full.task", "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"),
    ("https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
     "ssdlite320_mobilenet_v3_large_coco-a79551df.pth", "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"),
]


def main():
    directory = Path(__file__).resolve().parents[1] / "artifacts"
    directory.mkdir(exist_ok=True)
    for url, name, expected in MODELS:
        path = directory / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f"verified cache: {name}")
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            digest, size = hashlib.sha256(), 0
            with requests.get(url, stream=True, timeout=(20, 60)) as response, temporary.open("wb") as output:
                response.raise_for_status()
                for chunk in response.iter_content(1024*1024):
                    size += len(chunk)
                    if size > 64*1024*1024:
                        raise ValueError("模型下载超过预期大小")
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"{name} SHA256不匹配，保留原文件")
            temporary.replace(path)
            print(f"verified download: {name} ({size} bytes)")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
