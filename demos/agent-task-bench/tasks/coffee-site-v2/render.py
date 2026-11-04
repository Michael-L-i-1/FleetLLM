#!/usr/bin/env python3
"""Assemble the editable section fragments into one self-contained local site."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import tempfile

TASK = Path(__file__).resolve().parent
FILES = ['01-header.html', '02-hero.html', '03-mission.html', '04-menu.html',
         '05-impact.html', '06-events.html', '07-visit.html', '08-footer.html']


def render(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    sections = [(workspace/'sections'/name).read_text() for name in FILES]
    script = (TASK/'app.js').read_text()
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'sha256-" + digest + "'; base-uri 'none'; form-action 'none'"
    html = (TASK/'template.html').read_text().replace('{{csp}}', csp)
    html = html.replace('{{header}}', sections[0]).replace('{{main}}', '\n'.join(sections[1:7]))
    html = html.replace('{{footer}}', sections[7]).replace('{{script}}', script)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=output, suffix='.html', delete=False) as f:
        f.write(html)
        temporary = Path(f.name)
    temporary.replace(output/'index.html')
    return {'index': str(output/'index.html'), 'sha256': hashlib.sha256(html.encode()).hexdigest()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render(a.workspace, a.output)))
