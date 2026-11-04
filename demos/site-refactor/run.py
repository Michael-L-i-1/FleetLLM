#!/usr/bin/env python3
"""Engine-neutral coding demo. No model calls unless explicitly invoked with `run`."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
EDITABLE = {'index.html', 'styles.css', 'theme.css', 'responsive.css'}
FILES = EDITABLE | {'app.js'}
SUPERSEDED_OBSERVATION = json.dumps({'tool_result': 'Superseded file contents omitted. Re-read the current file if needed.'})
PROTOCOL = '''You are a coding agent. Reply with exactly one JSON object, no markdown.
Actions:
{"action":"read","path":"index.html"} (or styles.css, theme.css, responsive.css, app.js)
{"action":"write","path":"styles.css","content":"complete new file contents"}
{"action":"check"} runs the fixed browser and structure checks.
{"action":"done","summary":"what changed"}
Coordinator only, once: {"action":"delegate","tasks":[{"name":"layout","files":["index.html"],"task":"specific assignment"},{"name":"style","files":["styles.css"],"task":"specific assignment"},{"name":"theme","files":["theme.css"],"task":"specific assignment"},{"name":"responsive","files":["responsive.css"],"task":"specific assignment"}]}
Delegate exactly four tasks with one distinct editable file per worker. Workers run concurrently on separate
copies and their owned files are integrated after all workers succeed. Workers cannot delegate.
Only index.html, theme.css, styles.css, and responsive.css may change. Never remove existing behavior or use external assets.
Read files before editing. Tool responses are supplied as user messages. After delegation,
the coordinator reviews the merged files and fixes any issues. The coordinator must run check successfully before done.
'''


def parse_action(text):
    """Recover only a missing final object brace after a complete write payload.

    File bytes must already be a complete JSON string. No content, escaping,
    partial strings, arbitrary syntax, or model decisions are repaired.
    """
    try:
        return json.loads(text), False
    except json.JSONDecodeError as error:
        if error.pos != len(text) or not text.startswith('{'):
            raise
        try:
            action = json.loads(text + '}')
        except json.JSONDecodeError:
            raise error
        if (set(action) != {'action', 'path', 'content'} or action['action'] != 'write'
                or not isinstance(action['path'], str) or not isinstance(action['content'], str)):
            raise error
        return action, True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.cards = 0
        self.h1 = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        self.cards += 'data-workshop' in attrs
        self.h1 += tag == 'h1'


def check(site, output, browser=True):
    result = {'passed': False, 'checks': [], 'errors': []}
    try:
        if digest(site / 'app.js') != digest(HERE / 'fixture/app.js'):
            raise ValueError('app.js changed')
        source = (site / 'index.html').read_text()
        parsed = Structure()
        parsed.feed(source)
        required = {'home', 'workshops', 'signup', 'search', 'count', 'email', 'signup-form', 'signup-status'}
        if not required.issubset(parsed.ids) or len(parsed.ids) != len(set(parsed.ids)):
            raise ValueError('Missing or duplicate IDs')
        if parsed.cards != 3 or parsed.h1 != 1:
            raise ValueError('Keep three workshop cards and exactly one h1')
        for title in ['Intro to ceramics', 'Grow a kitchen garden', 'Print your own poster']:
            if title not in source:
                raise ValueError(f'Missing workshop: {title}')
        result['checks'].append('structure-and-preserved-script')
        if browser:
            proc = subprocess.run(['node', str(HERE / 'browser-check.mjs'), str(site), str(output)],
                                  capture_output=True, text=True, timeout=90)
            if proc.returncode:
                raise ValueError((proc.stderr or proc.stdout)[-4000:])
            result['checks'].append('browser-390-and-1440-search-signup-overflow-labels')
        result['passed'] = True
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result['errors'].append(str(error))
    return result


class Demo:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.site = self.root / 'site'
        shutil.copytree(HERE / 'fixture', self.site)
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.delegated = False
        self.calls = []
        self.actions = []
        self.context_drops = {}
        self.tool_errors = 0
        self.event('start', config=vars(args), fixture={f: digest(self.site / f) for f in sorted(FILES)})
        (self.root / 'brief.txt').write_text((HERE / 'brief.txt').read_text())

    def event(self, kind, **data):
        event = {'elapsed': round(time.monotonic() - self.started, 3), 'event': kind, **data}
        with self.lock:
            with (self.root / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
                stream.flush()
            print(json.dumps(event), flush=True)

    def request(self, name, messages):
        active_messages = messages
        if getattr(self.args, 'context_prune', False):
            # Drop superseded file observations only after a confirmed later write
            # (or after integration for the coordinator). Re-prefill the edited
            # history normally; no downstream KV is spliced across changed context.
            drop = set()
            reads = {}
            for i, msg in enumerate(messages[:-1]):
                if msg['role'] != 'assistant':
                    continue
                try:
                    action = json.loads(msg['content'])
                except (ValueError, TypeError):
                    continue
                if action.get('action') == 'read' and action.get('path') in EDITABLE and i+1 < len(messages)-1:
                    reads[action.get('path')] = i+1
                if action.get('action') in {'write', 'delegate'} and i+1 < len(messages):
                    try:
                        feedback = json.loads(messages[i+1]['content']).get('tool_result')
                    except (ValueError, TypeError):
                        continue
                    if action['action'] == 'write' and feedback == 'File written':
                        if action.get('path') in reads:
                            drop.add(reads.pop(action['path']))
                    elif action['action'] == 'delegate' and isinstance(feedback, list):
                        drop.update(reads.values())
                        reads.clear()
            drop = {i for i in drop if len(messages[i]['content']) >= getattr(self.args, 'context_prune_min_chars', 512)}
            if drop:
                active_messages = [{**m, 'content': SUPERSEDED_OBSERVATION} if i in drop else m
                                   for i,m in enumerate(messages)]
                self.event('context_edit', agent=name, masked_indices=sorted(drop),
                           dropped_chars=sum(len(messages[i]['content']) for i in drop),
                           original_messages=len(messages), active_messages=len(active_messages),
                           active_history_sha256=hashlib.sha256(json.dumps(active_messages,sort_keys=True).encode()).hexdigest())
        body = {'model': self.args.model, 'messages': active_messages, 'temperature': 0,
                'max_tokens': self.args.max_tokens, 'reasoning_effort': self.args.reasoning,
                'stream': True, 'stream_options': {'include_usage': True}}
        self.context_drops[name] = sorted(drop) if getattr(self.args, 'context_prune', False) else []
        req = urllib.request.Request(self.args.base.rstrip('/') + '/chat/completions',
                                     json.dumps(body).encode(), {'Content-Type': 'application/json'})
        start = time.monotonic()
        first = last = None
        with urllib.request.urlopen(req, timeout=self.args.timeout) as response:
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                reply = json.load(response)  # permits deterministic JSON fake servers
            else:
                content, reasoning, usage, finish, timings = [], [], {}, None, {}
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        break
                    chunk = json.loads(data)
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    if chunk.get('timings'):
                        timings = chunk['timings']
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta', {})
                        text = delta.get('content') or ''
                        thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if text or thought:
                            last = time.monotonic()
                            if first is None:
                                first = last
                                self.event('first_output', agent=name, seconds=first-start)
                        content.append(text)
                        reasoning.append(thought)
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
                reply = {'choices': [{'message': {'role': 'assistant', 'content': ''.join(content),
                         'reasoning_content': ''.join(reasoning)}, 'finish_reason': finish}],
                         'usage': usage, 'timings': timings}
        choice = reply['choices'][0]
        content = choice['message'].get('content') or ''
        seconds = time.monotonic() - start
        usage = reply.get('usage', {})
        call = {'agent': name, 'seconds': seconds, 'usage': usage,
                'finish_reason': choice.get('finish_reason'), 'ttft_seconds': first-start if first else None,
                'output_window_seconds': last-first if first and last else None,
                'request_completion_tokens_per_second': usage.get('completion_tokens', 0) / seconds,
                'timings': reply.get('timings', {})}
        with self.lock:
            self.calls.append(call)
        self.event('response', **call)
        return content, reply

    def delegate(self, tasks):
        if self.delegated:
            raise ValueError('Delegation has already been used')
        if not isinstance(tasks, list) or len(tasks) != 4:
            raise ValueError('Provide exactly four tasks')
        owned = set()
        for task in tasks:
            files = task.get('files', [])
            if len(files) != 1 or not set(files) <= EDITABLE or owned.intersection(files):
                raise ValueError('Each task needs nonoverlapping editable files')
            if not isinstance(task.get('task'), str) or not task['task'].strip():
                raise ValueError('Each worker needs a task')
            owned.update(files)
        self.delegated = True
        self.event('delegated', tasks=tasks)

        def worker(pair):
            i, task = pair
            name = f'worker-{i+1}'
            site = self.root / name / 'site'
            shutil.copytree(self.site, site)
            # Names are labels only; never use model-provided names as paths.
            summary = self.agent(name, site, set(task['files']), task['task'], coordinator=False)
            return task, site, summary

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            completed = list(pool.map(worker, enumerate(tasks)))
        for task, site, _ in completed:
            for file in task['files']:
                shutil.copyfile(site / file, self.site / file)
        self.event('integrated', files=sorted(owned))
        return [{'task': task['task'], 'summary': summary} for task, _, summary in completed]

    def agent(self, name, site, allowed, task, coordinator):
        protocol = PROTOCOL
        if not coordinator:
            owned = next(iter(allowed))
            protocol = PROTOCOL.split('Coordinator only, once:')[0]
            protocol = protocol.replace('"action":"write","path":"styles.css"',
                                        f'"action":"write","path":"{owned}"')
            protocol += (f'\nYou are one worker. Your only editable file is {owned}. '
                         'Read it, improve it for your assignment, write that file, then return done. '
                         'Other workers own the other files. The coordinator integrates all four files '
                         'and runs the final browser checks. Your copy contains the original versions '
                         'of other workers files. Keep app.js unchanged and use no external assets.\n')
        messages = [{'role': 'system', 'content': protocol + '\nShared brief:\n' + (HERE / 'brief.txt').read_text()},
                    {'role': 'user', 'content': f'Role: {name}. Editable files: {sorted(allowed)}. Task: {task}'}]
        passed = False
        consecutive_errors = 0
        transcript = self.root / f'{name}.jsonl'
        for step in range(self.args.max_steps):
            self.event('request', agent=name, step=step)
            content, reply = self.request(name, messages)
            with transcript.open('a') as stream:
                stream.write(json.dumps({'messages': messages, 'response': reply,
                                         'masked_message_indices': self.context_drops.get(name, [])}) + '\n')
            messages.append({'role': 'assistant', 'content': content})
            failed = False
            try:
                text = content.strip()
                if text.startswith('```'):
                    text = '\n'.join(text.splitlines()[1:-1])
                action, repaired = parse_action(text)
                if repaired:
                    self.event('protocol_repair', agent=name, repair='append-final-object-brace',
                               raw_sha256=hashlib.sha256(text.encode()).hexdigest())
                kind = action['action']
                self.event('action', agent=name, action=kind, path=action.get('path'))
                with self.lock:
                    self.actions.append({'agent': name, 'action': kind})
                if kind in {'read', 'write'}:
                    path = action.get('path')
                    if path not in FILES:
                        raise ValueError('Path must be an exact listed filename')
                    if kind == 'read':
                        result = (site / path).read_text()
                    else:
                        if path not in allowed:
                            raise ValueError('This file is not owned by you')
                        value = action['content']
                        if not isinstance(value, str) or len(value.encode()) > 100_000:
                            raise ValueError('File must be text under 100KB')
                        (site / path).write_text(value)
                        passed = False
                        result = 'File written'
                elif kind == 'delegate' and coordinator:
                    result = self.delegate(action['tasks'])
                    passed = False
                elif kind == 'check':
                    result = check(site, self.root / 'screenshots' / name)
                    passed = result['passed']
                elif kind == 'done':
                    if coordinator and (not passed or not self.delegated):
                        raise ValueError('Delegate work and pass checks after your last edit before finishing')
                    self.event('finished', agent=name, summary=action.get('summary', ''))
                    return action.get('summary', '')
                else:
                    raise ValueError('Unknown or unavailable action')
            except (ValueError, KeyError, TypeError) as error:
                with self.lock:
                    self.tool_errors += 1
                self.event('tool_error', agent=name, error=str(error))
                failed = True
                result = {'error': str(error)}
            consecutive_errors = consecutive_errors + 1 if failed else 0
            if getattr(self.args, 'max_consecutive_errors', 0) and consecutive_errors >= self.args.max_consecutive_errors:
                raise RuntimeError(f'{name} failed {consecutive_errors} consecutive tool actions')
            messages.append({'role': 'user', 'content': json.dumps({'tool_result': result})})
        raise RuntimeError(f'{name} exceeded {self.args.max_steps} steps')

    def run(self):
        success = False
        error = None
        try:
            before = check(self.site, self.root / 'screenshots/before')
            if not before['passed']:
                raise RuntimeError(f'Fixture/browser preflight failed: {before}')
            self.started = time.monotonic()  # Timing excludes fixture/browser setup, includes all agent/tool work.
            self.agent('coordinator', self.site, EDITABLE, 'Inspect the site and delegate exactly four workers: index.html (structure/accessibility), theme.css (colors/type/tokens), styles.css (components), responsive.css (layout/mobile). Integrate their work and verify the result.', True)
            final = check(self.site, self.root / 'screenshots/after')
            changed = [f for f in sorted(EDITABLE) if digest(self.site / f) != digest(HERE / 'fixture' / f)]
            success = final['passed'] and bool(changed)
            if not success:
                raise RuntimeError('Final checks failed or no files changed')
        except Exception as exc:
            error = str(exc)
            self.event('failed', error=error)
        finally:
            seconds = time.monotonic() - self.started
            for file in sorted(EDITABLE):
                diff = subprocess.run(['diff', '-u', str(HERE / 'fixture' / file), str(self.site / file)],
                                      capture_output=True, text=True).stdout
                (self.root / f'{file}.patch').write_text(diff)
            summary = {'label': self.args.label, 'model': self.args.model, 'success': success,
                       'seconds': seconds, 'calls': len(self.calls), 'error': error,
                       'prompt_tokens': sum(c['usage'].get('prompt_tokens', 0) for c in self.calls),
                       'completion_tokens': sum(c['usage'].get('completion_tokens', 0) for c in self.calls)}
            summary['aggregate_completion_tokens_per_second'] = summary['completion_tokens'] / max(seconds, 1e-9)
            summary['tool_errors'] = self.tool_errors
            summary['check_calls'] = sum(a['action'] == 'check' for a in self.actions)
            summary['write_calls'] = sum(a['action'] == 'write' for a in self.actions)
            ttfts = [c['ttft_seconds'] for c in self.calls if c['ttft_seconds'] is not None]
            summary['median_ttft_seconds'] = statistics.median(ttfts) if ttfts else None
            (self.root / 'calls.json').write_text(json.dumps(self.calls, indent=2))
            (self.root / 'summary.json').write_text(json.dumps(summary, indent=2))
            with (self.root / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=summary)
                writer.writeheader()
                writer.writerow(summary)
            self.event('complete', **summary)
        return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('check')
    validate.add_argument('--site', type=Path, default=HERE / 'fixture')
    validate.add_argument('--output', type=Path, default=HERE / 'runs/fixture-check')
    validate.add_argument('--structure-only', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('--base', default='http://127.0.0.1:8090/v1')
    run.add_argument('--model', default='glimmer-mlx4')
    run.add_argument('--output', required=True)
    run.add_argument('--label', required=True)
    run.add_argument('--max-steps', type=int, default=20)
    run.add_argument('--max-tokens', type=int, default=4096)
    run.add_argument('--reasoning', default='low', choices=['none','low','medium','high'])
    run.add_argument('--max-consecutive-errors', type=int, default=0, help='Stop a stalled agent after this many consecutive invalid actions (0: disabled)')
    run.add_argument('--context-prune-min-chars', type=int, default=512)
    run.add_argument('--context-prune', action='store_true', help='Opt-in removal of superseded tool observations; changes model context')
    run.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.command == 'check':
        result = check(args.site.resolve(), args.output.resolve(), not args.structure_only)
        print(json.dumps(result, indent=2))
        return 0 if result['passed'] else 1
    return Demo(args).run()


if __name__ == '__main__':
    raise SystemExit(main())
