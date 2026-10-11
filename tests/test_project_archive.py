import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('project_archive', Path(__file__).parents[1]/'scripts/project_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_restore_validates_before_writing_and_preserves_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, 'ROOT', tmp_path)
    directory = tmp_path/'archive'
    directory.mkdir()
    package = directory/'fixture.zip'
    contents = b'published model fixture'
    with zipfile.ZipFile(package, 'w') as output:
        output.writestr('artifacts/model.bin', contents)
    manifest = dict(archive='fixture.zip', archiveSha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                    targets=['artifacts'], directories=['artifacts/', 'artifacts/empty/'],
                    files=[dict(path='artifacts/model.bin', bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())])
    path = directory/'fixture.manifest.json'
    path.write_text(json.dumps(manifest))
    assert archive.restore('artifacts') == dict(target='artifacts', files=1, verified=True)
    assert (tmp_path/'artifacts/empty').is_dir()
    assert archive.restore('artifacts')['verified']
    (tmp_path/'artifacts/model.bin').write_bytes(b'locally changed')
    with pytest.raises(FileExistsError):
        archive.restore('artifacts')
    assert (tmp_path/'artifacts/model.bin').read_bytes() == b'locally changed'
    for invalid in ['../outside', '/outside', 'C:/outside', 'artifacts/../../outside']:
        with pytest.raises(ValueError):
            archive.restore(invalid)
    manifest['files'][0]['sha256'] = '0'*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Entry SHA256 mismatch'):
        archive.restore('artifacts')
