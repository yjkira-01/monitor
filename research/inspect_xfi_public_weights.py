"""Read only author-linked public Drive folder metadata; no account credentials."""

import json
import http.client
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/paper_baseline_sources_20261013/public_weight_metadata"


def folder(folder_id):
    url = "https://drive.google.com/drive/folders/" + folder_id
    html = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Connection": "close"}), timeout=30) as response:
                data = response.read()
                if len(data) > 4 * 1024 ** 2:
                    raise ValueError("Public folder response exceeds metadata limit")
                html = data.decode("utf-8")
            break
        except http.client.IncompleteRead:
            if attempt == 1:
                raise
    match = re.search(r"window\['_DRIVE_ivd'\] = '(.+?)';", html)
    if not match:
        raise ValueError("Public folder metadata not present; no authentication attempted")
    decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), match[1]).replace(r"\/", "/")
    listing = json.loads(decoded)[0]
    rows = [{"id": value[0], "name": value[2], "mime": value[3]} for value in listing or []]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / (folder_id + ".html")).write_text(html, encoding="utf-8")
    print(json.dumps({"public_folder": url, "entries": rows}), flush=True)
    return rows


def main():
    todo = [("1NUkOtaDR08JY5G-rxC-uUuMY6rs_G_rJ", "MMFi_HPE", 0)]
    result = []
    while todo:
        identifier, path, depth = todo.pop(0)
        children = folder(identifier)
        for child in children:
            child["path"] = path + "/" + child["name"]
            result.append(child)
            if child["mime"] == "application/vnd.google-apps.folder" and depth < 3:
                name = child["name"].lower()
                if depth == 0 or any(x in name for x in ("rgb", "mmwave", "backbone")):
                    todo.append((child["id"], child["path"], depth + 1))
    (OUTPUT / "public_weights_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
