#!/usr/bin/env python3
"""Package model-free traces and verify every archived file by SHA-256."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ARCHIVES=HERE/'archives'
ARCHIVES.mkdir(exist_ok=True)

def sha(data):return hashlib.sha256(data).hexdigest()
def archive(name, paths):
    target=ARCHIVES/name
    manifest={}
    with target.open('wb') as out:
        with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w') as tar:
                for path in sorted(set(paths)):
                    data=path.read_bytes();relative=str(path.relative_to(HERE))
                    manifest[relative]={'sha256':sha(data),'bytes':len(data)}
                    info=tarfile.TarInfo(relative);info.size=len(data);info.mtime=0;info.mode=0o644
                    tar.addfile(info,io.BytesIO(data))
    with tarfile.open(target,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==len(manifest)
        for member in members:
            data=tar.extractfile(member).read()
            assert sha(data)==manifest[member.name]['sha256'],member.name
    return {'file':str(target.relative_to(HERE)),'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'files':manifest}

# No weights, credentials, developer-provider logs, node_modules or generated binaries.
paths=[]
for folder in ['comparison','historical','excluded','smoke-fleet','smoke-vllm','smoke-vllm-plugin']:
    for p in (HERE/folder).rglob('*'):
        if not p.is_file() or p.suffix.lower() in ['.png','.mp4','.jpg','.webp']:continue
        if p.name=='auth.json' or '/pi-agent/' in str(p):continue
        if '/capture/preview/' in str(p):continue
        paths.append(p)
paths.extend(p for p in (HERE/'setup/single-request-capture').iterdir() if p.is_file())
records=[archive('traces.tar.gz',paths)]
# Original screenshots stay local and are separately hashed; the final video and
# exported websites are checked in directly. No sampled-frame substitutions.
images=[p for folder in ['comparison','historical','excluded'] for p in (HERE/folder).rglob('*.png')]
records.append(archive('captures.local.tar.gz',images))
manifest={'archives':records,'note':'Trace archive is portable and checked in. Full capture archive stays local because of size; its file hashes preserve screenshot provenance. Final video, poster, websites and summary are checked in directly.'}
(HERE/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='files'} for r in records],indent=2))
