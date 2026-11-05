"""Deterministic fake-model integration test; never loads a model."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

import run


class HarnessTest(unittest.TestCase):
    def test_delegation_merge_and_path_rejection(self):
        requests = []
        barrier = threading.Barrier(4)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                role = body['messages'][1]['content'].split('.')[0]
                turn = sum(m['role'] == 'assistant' for m in body['messages'])
                if 'coordinator' in role:
                    actions = [
                        {'action': 'read', 'path': '../../README.md'},
                        {'action': 'delegate', 'tasks': [
                            {'name': '../escape', 'files': ['index.html'], 'task': 'Improve layout'},
                            {'name': 'style', 'files': ['styles.css'], 'task': 'Improve style'},
                            {'name': 'theme', 'files': ['theme.css'], 'task': 'Improve theme'},
                            {'name': 'responsive', 'files': ['responsive.css'], 'task': 'Improve layout'}]},
                        {'action': 'check'}, {'action': 'done', 'summary': 'Integrated'}]
                    action = actions[turn]
                else:
                    file = ['index.html', 'styles.css', 'theme.css', 'responsive.css'][int(role.split('worker-')[1]) - 1]
                    if turn == 0:
                        barrier.wait(timeout=5)  # proves real concurrent worker requests
                        value = (run.HERE / 'fixture' / file).read_text()
                        value += '\n<!-- layout -->' if file.endswith('html') else '\nbody { color: #123; }'
                        action = {'action': 'write', 'path': file, 'content': value}
                    else:
                        action = {'action': 'done', 'summary': 'Updated owned file'}
                reply = {'choices': [{'message': {'content': json.dumps(action)}, 'finish_reason': 'stop'}],
                         'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(output=str(Path(tmp) / 'trial'), model='fake', label='test',
                                          base=f'http://127.0.0.1:{server.server_port}/v1',
                                          max_steps=8, max_tokens=100, reasoning='low', timeout=10)
                original = run.check
                with patch.object(run, 'check', side_effect=lambda site, output: original(site, output, browser=False)):
                    demo = run.Demo(args)
                    self.assertEqual(demo.run(), 0)
                self.assertTrue((demo.root / 'summary.csv').exists())
                self.assertIn('layout', (demo.site / 'index.html').read_text())
                self.assertIn('color: #123', (demo.site / 'styles.css').read_text())
                self.assertFalse((Path(tmp) / 'escape').exists())
                self.assertEqual(len(requests), 12)
                self.assertIn('error', requests[1]['messages'][-1]['content'])
                self.assertEqual(run.digest(demo.site / 'app.js'), run.digest(run.HERE / 'fixture/app.js'))
                with self.assertRaises(ValueError):
                    demo.delegate([])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_behavior_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / 'site'
            run.shutil.copytree(run.HERE / 'fixture', site)
            (site / 'app.js').write_text('')
            self.assertFalse(run.check(site, Path(tmp), browser=False)['passed'])


if __name__ == '__main__':
    unittest.main()
