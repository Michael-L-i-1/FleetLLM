#!/usr/bin/env python3
"""Cumulative festival checks for build, stakeholder update and browser polish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from render import render

TASK = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--round', type=int, choices=[1,2,3], default=3)
a = p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
try:
    built = render(a.workspace,a.output/'site')
    command = ['node',str(TASK/'browser-check.mjs'),str(a.workspace.resolve()),
               str(Path(built['index']).parent),str(a.output.resolve()),str(a.round)]
    process = subprocess.run(command,capture_output=True,text=True,timeout=120)
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        result = {'passed':False,'failures':['Browser check did not return JSON'],'stdout':process.stdout,'stderr':process.stderr}
    result['browser_exit_code'] = process.returncode
    result['render'] = built
except Exception as error:
    result = {'passed':False,'failures':[f'{type(error).__name__}: {error}']}
result['round'] = a.round
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
sys.exit(0 if result['passed'] else 1)
