"""下载官方 URFD camera 0 RGB 序列；保留原始 ZIP，独立于训练数据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from urllib.parse import urljoin
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"


def download(url, directory):
    name = url.rsplit("/", 1)[1]
    if not re.fullmatch(r"(?:adl|fall)-\d{2}-cam0-rgb\.zip", name):
        raise ValueError("不支持的 URFD 文件名")
    path = directory / name
    if not path.exists():
        temporary = path.with_suffix(".zip.part")
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45) as response, temporary.open("wb") as output:
                expected = int(response.headers.get("Content-Length", "0"))
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 512 * 1024 * 1024:
                        raise ValueError("序列超出 512MiB 限制")
                    output.write(chunk)
                if not size or expected and size != expected:
                    raise ValueError("下载长度不完整")
            with zipfile.ZipFile(temporary) as archive:
                if not any(n.lower().endswith(".png") for n in archive.namelist()):
                    raise ValueError("下载文件没有 PNG RGB 帧")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(path) as archive:
        frames = [n for n in archive.namelist() if n.lower().endswith(".png")]
        if not frames:
            raise ValueError("缓存文件没有 RGB 帧")
        archive.read(frames[0])
        archive.read(frames[-1])
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024*1024):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "split": "external", "label": name.split("-")[0],
            "dataset": "URFD", "sequence": name.removesuffix("-cam0-rgb.zip"), "camera": "0", "frames": len(frames),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "url": url,
            "license": "CC BY-NC-SA 4.0", "subject": "unknown"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=40, help="每类按编号取前 N 个；默认取全部70序列")
    args = parser.parse_args()
    if not 1 <= args.per_class <= 40:
        parser.error("per-class 必须为 1–40")
    directory = ROOT / "data/raw/URFD"
    directory.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(PAGE, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as response:
        html = response.read().decode("utf-8")
    links = sorted(set(re.findall(r'href\s*=\s*["\x27]\s*([^"\x27]*cam0-rgb.zip)', html, re.I)))
    selected = [urljoin(PAGE, name) for label in ("adl", "fall") for name in [x for x in links if f"/{label}-" in x][:args.per_class]]
    if len(links) != 70 or not selected:
        raise ValueError("官方页面序列数发生变化，请重新核对下载链接")
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, url, directory): url for url in selected}
        for future in as_completed(jobs):
            try:
                row = future.result()
                rows.append(row)
                print(f"downloaded {len(rows)}/{len(selected)} {row['sequence']} frames={row['frames']}", flush=True)
            except Exception as exc:
                errors.append({"url": jobs[future], "error": str(exc)})
                print(f"failed {jobs[future]}: {exc}", flush=True)
    rows.sort(key=lambda r: r["sequence"])
    manifest = ROOT / "data/manifests/urfd_external.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "artifacts/urfd_download_report.json").write_text(json.dumps({"source": PAGE, "rows": rows, "errors": errors}, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"{len(errors)} 个序列未下载成功，可再次执行以保留已有缓存")
    assert len(rows) == len(selected)


if __name__ == "__main__":
    main()
