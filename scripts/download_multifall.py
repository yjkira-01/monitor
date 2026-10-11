"""Download checksum-verified temporal annotations and videos from Zenodo 17170592."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen
from urllib.request import Request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RECORD = "https://zenodo.org/api/records/17170592"


def fetch_part(item, directory, start):
    end = min(start + 8 * 1024**2, item["size"]) - 1
    path = directory / f"{item['key']}.{start}.chunk"
    length = end - start + 1
    if path.exists() and path.stat().st_size == length:
        return path
    for attempt in range(4):
        try:
            request = Request(item["links"]["self"], headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=120) as response, path.open("wb") as output:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{item['size']}":
                    raise ValueError("Server did not honor byte range")
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > length:
                        raise ValueError("Range exceeded expected length")
                    output.write(chunk)
            if path.stat().st_size != length:
                raise ValueError("Incomplete byte range")
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def extract(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive entry: {info.filename}")
        archive.extractall(destination)


def download(item, directory):
    path = directory / item["key"]
    temporary = path.with_suffix(".zip.part")
    expected = item["checksum"].removeprefix("md5:")
    for attempt in range(4):
        try:
            digest = hashlib.md5()
            if path.exists():
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
            else:
                starts = range(0, item["size"], 8 * 1024**2)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    chunks = list(pool.map(lambda start: fetch_part(item, directory, start), starts))
                with temporary.open("wb") as handle:
                    for part in chunks:
                        with part.open("rb") as source:
                            while chunk := source.read(1024 * 1024):
                                digest.update(chunk)
                                handle.write(chunk)
                if temporary.stat().st_size != item["size"]:
                    raise ValueError("Incomplete archive")
            if digest.hexdigest() != expected:
                raise ValueError("Archive checksum mismatch")
            if not path.exists():
                temporary.replace(path)
            destination = directory / path.stem
            extract(path, destination)
            for part in directory.glob(f"{item['key']}.*.chunk"):
                part.unlink()
            print(f"verified/extracted {item['key']} bytes={item['size']}", flush=True)
            return {"file": item["key"], "bytes": item["size"], "md5": expected, "source": item["links"]["self"]}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=["Cauca_fall.zip", "UP_Fall.zip"])
    parser.add_argument('--project-root',default=str(ROOT))
    parser.add_argument('--report',default='artifacts/multifall_download.json')
    args = parser.parse_args()
    ROOT = Path(args.project_root).resolve()
    with urlopen(RECORD, timeout=45) as response:
        metadata = json.load(response)
    files = {item["key"]: item for item in metadata["files"]}
    if any(name not in files for name in args.files):
        parser.error("Unknown archive")
    directory = ROOT / "data/raw/OmniFall"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: download(files[name], directory), args.files))
    report = {"record": RECORD, "license": metadata["metadata"].get("license"), "files": results}
    (ROOT / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
