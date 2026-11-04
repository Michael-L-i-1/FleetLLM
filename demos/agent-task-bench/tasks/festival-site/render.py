#!/usr/bin/env python3
"""Deterministically assemble eight editable fragments into eight local pages."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import tempfile

TASK = Path(__file__).resolve().parent
PAGES = {'home':'Home','lineup':'Lineup','schedule':'Schedule','venues':'Venues',
         'tickets':'Tickets','accessibility':'Accessibility','volunteer':'Volunteer','faq':'FAQ'}
OUTPUTS = {name: 'index.html' if name == 'home' else name+'.html' for name in PAGES}


def atomic_write(path, data):
    if path.exists() and path.read_bytes() == data:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        file.write(data)
        temporary = Path(file.name)
    temporary.replace(path)


def render(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'assets').mkdir(exist_ok=True)
    atomic_write(output/'assets/lantern.svg', (TASK/'assets/lantern.svg').read_bytes())
    script = (TASK/'app.js').read_text()
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = "default-src 'none'; style-src 'unsafe-inline'; img-src 'self' file: data:; script-src 'sha256-" + digest + "'; base-uri 'none'; form-action 'none'"
    template = (TASK/'template.html').read_text()
    hashes = {}
    for name, label in PAGES.items():
        nav = ''.join('<a href="'+OUTPUTS[key]+'"'+(' aria-current="page"' if key==name else '')+'>'+html.escape(value)+'</a>' for key,value in PAGES.items())
        body = template.replace('{{csp}}', csp).replace('{{title}}', label).replace('{{nav}}', nav)
        body = body.replace('{{fragment}}', (workspace/'pages'/f'{name}.html').read_text()).replace('{{script}}', script)
        data = body.encode()
        atomic_write(output/OUTPUTS[name], data)
        hashes[OUTPUTS[name]] = hashlib.sha256(data).hexdigest()
    return {'index':str(output/'index.html'), 'sha256':hashes['index.html'], 'pages':list(OUTPUTS.values()),
            'source_to_output':{'pages/'+key+'.html':value for key,value in OUTPUTS.items()}, 'page_sha256':hashes}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render(a.workspace,a.output)))
