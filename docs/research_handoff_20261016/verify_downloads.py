"""Verify downloaded release assets using SHA256SUMS.txt (standard library).

By default verify the files present, allowing the optional history bundle to be
omitted. --all requires every listed asset. No files are modified or removed.
"""
import argparse,hashlib,re
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--all',action='store_true',dest='require_all');args=parser.parse_args()
    root=args.directory.resolve();passed=[];missing=[]
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        expected,name=line.split('  ',1);assert re.fullmatch('[0-9a-f]{64}',expected)
        path=(root/name).resolve();assert root in path.parents,'Manifest path escapes download directory'
        if not path.is_file():missing.append(name);continue
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==expected,'SHA256 mismatch: '+name
        passed.append(name);print('PASS',name)
    if args.require_all and missing:raise SystemExit('Missing files: '+', '.join(missing))
    assert passed,'No downloaded files were found'
    print('Verified files:',len(passed),'Not downloaded:',len(missing))


if __name__=='__main__':main()
