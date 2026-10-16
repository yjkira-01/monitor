"""Retrieve the author-linked public HPE ZIP and extract only two state dicts."""

import hashlib
import json
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/paper_baseline_sources_20261013/xfi_pretrained"
IDENTIFIER = "1qkMa1vol8w55QvytbeKxzOSkKfwrIpGu"
LIMIT = 768 * 1024 ** 2


class PublicDownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active, self.action, self.inputs = False, None, {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "form" and values.get("id") == "download-form":
            self.active, self.action = True, values.get("action")
        if tag == "input" and self.active and "name" in values:
            self.inputs[values["name"]] = values.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def main():
    if OUTPUT.exists():
        raise ValueError("New output required; no overwrite of pretrained assets")
    OUTPUT.mkdir(parents=True)
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode({"id": IDENTIFIER, "export": "download"})
    response = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45)
    if "text/html" in response.headers.get("Content-Type", ""):
        html = response.read(1024 ** 2).decode("utf-8")
        response.close()
        (OUTPUT / "public_download_confirmation.html").write_text(html, encoding="utf-8")
        form = PublicDownloadForm()
        form.feed(html)
        if (form.action != "https://drive.usercontent.google.com/download"
                or form.inputs.get("id") != IDENTIFIER):
            raise ValueError("Public standard download form absent; no login or access workaround attempted")
        url = form.action + "?" + urllib.parse.urlencode(form.inputs)
        response = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60)
    if "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError("Public file unavailable; stop without authentication")
    destination = OUTPUT / "backbones.zip"
    digest, total, last_print = hashlib.sha256(), 0, 0
    with response, destination.open("xb") as stream:
        while chunk := response.read(4 * 1024 ** 2):
            total += len(chunk)
            if total > LIMIT:
                raise ValueError("Public backbone archive exceeds bounded download size")
            stream.write(chunk)
            digest.update(chunk)
            if total - last_print >= 32 * 1024 ** 2:
                print(json.dumps({"downloaded_MB": round(total / 1024 ** 2, 1)}), flush=True)
                last_print = total
    selected = {}
    with zipfile.ZipFile(destination) as archive:
        for name in ("RGB_Resnet18.pt", "mmwave_all_random_TD.pt"):
            matches = [row for row in archive.infolist() if Path(row.filename).name == name and not row.is_dir()]
            if len(matches) != 1 or matches[0].file_size > 256 * 1024 ** 2:
                raise ValueError("Unique bounded native HPE state dict not found: " + name)
            value = archive.read(matches[0])
            (OUTPUT / name).write_bytes(value)
            selected[name] = {"zip_member": matches[0].filename, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    receipt = {"status": "complete", "source_repository": "https://github.com/NTUMARS/X-Fi",
               "author_linked_public_folder": "https://drive.google.com/drive/folders/1ShcQqUd5RnqsTBZ3yM97hrooa7p06O2g",
               "public_HPE_folder": "https://drive.google.com/drive/folders/1NUkOtaDR08JY5G-rxC-uUuMY6rs_G_rJ",
               "public_file_id": IDENTIFIER, "archive_bytes": total, "archive_sha256": digest.hexdigest(),
               "selected_state_dicts": selected, "serialization_loaded": False,
               "scope": "Author-provided MM-Fi HPE pretrained RGB and mmWave backbones; no mRI scores"}
    (OUTPUT / "fetch_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "archive_MB": total / 1024 ** 2, "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
