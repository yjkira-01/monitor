"""Archive bounded public official source snapshots without executing them."""

import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/paper_baseline_sources_20261013"
REPOSITORIES = {"awc": "MICLAB-BUPT/AWC"}


def fetch(url, limit=64 * 1024 ** 2):
    request = urllib.request.Request(url, headers={"User-Agent": "Research-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Public response exceeds bounded source limit")
        return data, response.status, response.url


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, repo in REPOSITORIES.items():
        destination = DEST / name
        if destination.exists():
            raise ValueError("Refusing to overwrite source archive: " + str(destination))
        listing = subprocess.run(["git", "ls-remote", "https://github.com/" + repo + ".git", "refs/heads/main"],
                                 check=True, capture_output=True, text=True, timeout=45).stdout.strip()
        commit = listing.split()[0]
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("Invalid public commit identity")
        data, status, url = fetch("https://codeload.github.com/" + repo + "/zip/" + commit)
        destination.mkdir()
        (destination / "official_source.zip").write_bytes(data)
        manifest = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(row.file_size for row in archive.infolist()) > 128 * 1024 ** 2:
                raise ValueError("Uncompressed source exceeds limit")
            for row in archive.infolist():
                if row.is_dir():
                    continue
                relative = PurePosixPath(*PurePosixPath(row.filename).parts[1:])
                if relative.is_absolute() or ".." in relative.parts or any(":" in p for p in relative.parts):
                    raise ValueError("Unsafe source path")
                contents = archive.read(row)
                digest = hashlib.sha256(contents).hexdigest()
                manifest.append({"path": str(relative), "bytes": len(contents), "sha256": digest})
                if relative.suffix.lower() in (".py", ".md", ".txt", ".yaml", ".yml", ".json", ".sh") or relative.name == "LICENSE":
                    target = destination / "source" / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(contents)
        receipt = {"status": "complete", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                   "repository": "https://github.com/" + repo, "commit": commit,
                   "archive_url": url, "http_status": status, "archive_bytes": len(data),
                   "archive_sha256": hashlib.sha256(data).hexdigest(), "files": manifest,
                   "source_executed": False}
        (destination / "fetch_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({"repository": repo, "commit": commit, "files": len(manifest),
                          "python_files": sum(p["path"].endswith(".py") for p in manifest), "archive_bytes": len(data)}), flush=True)


if __name__ == "__main__":
    main()
