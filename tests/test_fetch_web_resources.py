import hashlib
import importlib
import io
import json
from pathlib import Path
import zipfile

import pytest


def test_download_checks_hash_before_restoring(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    fetch = importlib.import_module('fetch_web_resources')
    archive = importlib.import_module('project_archive')
    monkeypatch.setattr(fetch, 'ROOT', tmp_path)
    monkeypatch.setattr(archive, 'ROOT', tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as output:
        output.writestr('artifacts/test.pt', b'model')
    contents = buffer.getvalue()
    manifest = dict(url='https://github.com/yjkira-01/monitor/releases/download/v0.3.2/web-resources.zip',
                    archive='web-resources.zip', archiveBytes=len(contents),
                    archiveSha256=hashlib.sha256(contents).hexdigest(), targets=['artifacts'], directories=[],
                    files=[dict(path='artifacts/test.pt', bytes=5, sha256=hashlib.sha256(b'model').hexdigest())])
    path = tmp_path/'web-resources.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'x'*len(contents)))
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch.main()
    assert not (tmp_path/'artifacts/test.pt').exists()
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(contents))
    fetch.main()
    assert (tmp_path/'artifacts/test.pt').read_bytes() == b'model'
    monkeypatch.setattr(fetch.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Valid cache must not download'))
    fetch.main()
    manifest['archive'] = '../outside.zip'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        fetch.main()
