#!/usr/bin/env python3
"""Build and check the real browser behavior of the eight-section site."""
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
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
try:
    built = render(a.workspace, a.output/'site')
    command = ['node', str(TASK/'browser-check.mjs'), str(a.workspace.resolve()),
               str(Path(built['index']).parent), str(a.output.resolve())]
    process = subprocess.run(command, capture_output=True, text=True, timeout=90)
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        result = {'passed': False, 'failures': ['Browser check did not return JSON'],
                  'stdout': process.stdout, 'stderr': process.stderr}
    result['browser_exit_code'] = process.returncode
    result['render'] = built
except Exception as error:
    result = {'passed': False, 'failures': [f'{type(error).__name__}: {error}']}
(a.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
sys.exit(0 if result['passed'] else 1)
