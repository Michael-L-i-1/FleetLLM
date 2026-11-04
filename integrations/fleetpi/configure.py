#!/usr/bin/env python3
"""Prepare the optional local vLLM baseline with model-correct Glimmer attention."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vllm-bin', default=shutil.which('vllm'))
parser.add_argument('--model-path', type=Path, default=Path.home()/'.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-4bit/snapshots/3e7677d7a40d348a3daba263a2b1c0aa41910710')
args = parser.parse_args()
if not args.vllm_bin or not args.model_path.is_dir():
    raise SystemExit('Installed vLLM-Metal and the pinned Glimmer four-bit snapshot are required')
vllm = Path(args.vllm_bin).resolve()
shebang = vllm.read_text().splitlines()[0]
if not shebang.startswith('#!/'):
    raise SystemExit('Cannot identify the installed vLLM Python; use its direct executable')
python = shebang[2:]
source = subprocess.check_output([python, '-c', 'import importlib.util;print(next(iter(importlib.util.find_spec("vllm_metal").submodule_search_locations)))'], text=True).strip()
overlay = ROOT/'build/task-vllm-overlay'
if not overlay.exists():
    subprocess.run(['python3', str(ROOT/'demos/agent-task-bench/compat/prepare.py'), '--source', source, '--output', str(overlay)], check=True)
# prepare.py checks source/version hashes and keeps the global installation untouched.
config = {'engine':'vllm-metal-default', 'model':'glimmer-mlx4', 'base':'http://127.0.0.1:8090/v1',
 'command':[str(vllm),'serve',str(args.model_path.resolve()),'--served-model-name','glimmer-mlx4','--host','127.0.0.1','--port','8090','--reasoning-parser','muse_glimmer','--language-model-only','--enable-auto-tool-choice','--tool-call-parser','muse_glimmer'],
 'env':{'PYTHONPATH':str(overlay)},
 'unset_env':['VLLM_METAL_COMPILED_MLP','VLLM_ENABLE_V1_MULTIPROCESSING','VLLM_METAL_MULTIMODAL_MODE','VLLM_USE_V2_MODEL_RUNNER','VLLM_METAL_USE_NAX'],
 'baseline_contract':'Installed vLLM-Metal 0.32.0 performance defaults. Default prefix cache and batching remain enabled. Isolated Glimmer correctness overlay and official native tool/reasoning parsers are required compatibility fixes, not performance handicaps.'}
path = ROOT/'integrations/fleetpi/configs/vllm-default.local.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(config,indent=2)+'\n')
print(path)
