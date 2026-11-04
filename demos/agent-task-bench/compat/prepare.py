#!/usr/bin/env python3
"""Create an isolated vLLM-Metal0.32.0 Glimmer compatibility overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True,help='Installed vllm_metal package directory')
p.add_argument('--output',type=Path,required=True,help='Fresh overlay PYTHONPATH directory')
a=p.parse_args()
hashes=json.loads((HERE/'source-hashes.json').read_text())
for name,expected in hashes.items():
    assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==expected['original_sha256'], f'Source mismatch: {name}'
a.output.mkdir(parents=True,exist_ok=False)
package=a.output/'vllm_metal'
shutil.copytree(a.source,package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
subprocess.run(['git','apply','--check',str(HERE/'vllm-glimmer.patch')],cwd=package,check=True)
subprocess.run(['git','apply',str(HERE/'vllm-glimmer.patch')],cwd=package,check=True)
for name,expected in hashes.items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected['patched_sha256']
print(json.dumps({'overlay':str(a.output.resolve()),'patched_files':list(hashes)}))
