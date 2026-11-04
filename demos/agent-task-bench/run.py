#!/usr/bin/env python3
"""Real coding agents: shared tools, protected checks, durable results; no scripted solutions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
class InfrastructureError(RuntimeError):
    pass
PROTOCOL = '''You are a software engineering agent editing a local project. Complete the assigned work.
Return exactly ONE JSON object per response, without Markdown or surrounding prose.
Tools:
{"action":"read","path":"relative/file"}
{"action":"write","path":"relative/file","content":"complete UTF-8 file contents"}
{"action":"check"} -- integration/single agent only; runs the protected acceptance suite.
{"action":"done","summary":"brief description"}
Use the exact action/path/content keys. Source code belongs INSIDE the JSON content string.
Only your listed editable files may be written. Never edit tests or protected infrastructure.
Do not add dependencies, external assets, or network calls. Implement behavior, not test bypasses.
Workers own disjoint files in the same workspace; other workers may be changing their files.
Workers should implement their owned file and finish; the harness runs integration checks after
all workers finish and sends failures to an integration agent if repairs are needed.
The check tool is unavailable to workers, because other modules may still be incomplete.
Tool feedback arrives as a user message. The workspace is authoritative; reread if uncertain.
Keep edits concise enough to fit one complete JSON response. Never claim an unperformed check.
'''


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def parse_action(content):
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Return one JSON object')
    return value


class Benchmark:
    def __init__(self, args):
        self.a = args
        self.task_dir = args.task.resolve()
        self.spec = json.loads((self.task_dir/'task.json').read_text())
        defaults = {'max_calls':108, 'worker_calls':3, 'integration_calls':12, 'round_calls':36,
                    'round_wall_limit':800, 'token_budget':96000, 'wall_limit':2700} if self.spec.get('rounds') else {
                    'max_calls':56, 'worker_calls':6, 'integration_calls':56, 'round_calls':56,
                    'round_wall_limit':2700, 'token_budget':96000, 'wall_limit':2700}
        for key,value in defaults.items():
            if getattr(args,key,None) is None:setattr(args,key,self.spec.get('harness',{}).get(key,value))
        self.rounds = self.spec.get('rounds') or [{'id':1,'title':'Task','instructions':''}]
        if [r['id'] for r in self.rounds] != list(range(1,len(self.rounds)+1)):
            raise ValueError('Round ids must be consecutive integers starting at 1')
        workers=self.spec.get('workers',self.spec.get('worker_assignments',[]))
        if len({w['name'] for w in workers}) != len(workers) or len({w.get('file',w.get('path')) for w in workers}) != len(workers):
            raise ValueError('Worker names and editable ownership must be disjoint')
        self.current_round = 0
        self.round_title = ''
        self.phase = 'setup'
        self.round_started = None
        self.round_call_start = 0
        self.histories = {}
        self.round_results = []
        self.round_first_pass = {}
        self.cancelled = threading.Event()
        self.active_responses = set()
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.workspace = self.out/'workspace'
        shutil.copytree(self.task_dir/self.spec.get('fixture', 'fixture'), self.workspace)
        self.editable = set(self.spec['editable_files'])
        self.protected = {str(p.relative_to(self.workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in self.workspace.rglob('*') if p.is_file() and str(p.relative_to(self.workspace)) not in self.editable}
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.start_epoch = time.time()
        self.call_count = self.tokens = self.tool_errors = self.protocol_errors = 0
        self.calls = []
        self.worker_results = []
        self.check_count = 0
        self.first_pass_seconds = None
        self.capture = None
        (self.out/'responses').mkdir()
        self.shared = self.make_shared()
        self.event('start', config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                   started_unix=self.start_epoch, task_sha256=hashlib.sha256((self.task_dir/'task.json').read_bytes()).hexdigest(),
                   initial_files={str(p.relative_to(self.workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in self.workspace.rglob('*') if p.is_file()})

    def event(self, kind, **fields):
        with self.lock:
            record = {'event': kind, 'elapsed': time.monotonic()-self.started, 'unix_seconds': time.time(), 'round':self.current_round, 'round_title':self.round_title, 'phase':self.phase, **fields}
            with (self.out/'events.jsonl').open('a') as out:
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
                out.flush()
        return record

    def make_shared(self):
        brief = self.spec.get('instructions', '')
        if self.spec.get('brief_file'):
            brief += '\n'+(self.workspace/self.spec['brief_file']).read_text()
        # A single identical prefix supplies contracts and initial state to every worker.
        sources = []
        for p in sorted(self.workspace.rglob('*')):
            if p.is_file() and p.suffix in {'.md','.rs','.c','.h','.html','.css','.js','.json'} and p.stat().st_size < 24000:
                sources.append(f'\n--- {p.relative_to(self.workspace)} ---\n{p.read_text()}')
        return 'PROJECT TASK\n'+brief+'\n\nINITIAL PROJECT SNAPSHOT (subsequent tool reads are authoritative)\n'+''.join(sources)

    def path(self, name):
        if not isinstance(name, str) or '\\' in name:
            raise ValueError('Path must be a relative file name')
        path = self.workspace/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(self.workspace):
            raise ValueError('Path leaves workspace')
        return path

    def compact(self, messages, agent):
        if sum(len(x['content']) for x in messages) <= self.a.max_prompt_chars:
            return messages
        recent = list(messages[2:])
        dropped = 0
        # Keep stakeholder instructions from every round; omit oldest tool exchanges only.
        while sum(len(x['content']) for x in messages[:2]+recent) > self.a.max_prompt_chars-500:
            index=next((i for i,m in enumerate(recent[:-2]) if m['role']=='assistant'),None)
            if index is None:break
            count=1
            if index+1<len(recent) and recent[index+1]['role']=='user' and not recent[index+1]['content'].startswith('CURRENT ROUND'):
                count=2
            del recent[index:index+count]
            dropped+=count
        notice='\nEarlier tool exchanges were omitted to bound context. Keep all round requirements above. '
        notice+='Read the current workspace before changing an older file.'
        value = messages[:1]+[{**messages[1],'content':messages[1]['content']+notice}]+recent
        self.event('context_compaction', agent=agent, dropped_messages=dropped,
                   original_chars=sum(len(x['content']) for x in messages), sent_chars=sum(len(x['content']) for x in value))
        return value

    def cancel(self, reason='Cancellation requested'):
        self.cancel_reason = reason
        self.cancelled.set()
        with self.lock:responses = list(self.active_responses)
        self.interrupt_responses(responses)
        self.event('cancel',reason=reason)

    def interrupt_responses(self, responses):
        for response in responses:
            try:response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError,OSError):pass

    def deadline_watch(self, done):
        while not done.wait(.1):
            if self.remaining_seconds() <= 0 or self.cancelled.is_set():
                with self.lock:responses=list(self.active_responses)
                self.interrupt_responses(responses)

    def remaining_seconds(self):
        remaining = self.a.wall_limit-(time.monotonic()-self.started)
        if self.round_started is not None:
            remaining=min(remaining,self.a.round_wall_limit-(time.monotonic()-self.round_started))
        return remaining

    def request(self, agent, messages):
        with self.lock:
            if self.cancelled.is_set() or self.call_count >= self.a.max_calls or self.tokens >= self.a.token_budget or self.remaining_seconds() <= 0 or self.call_count-self.round_call_start >= self.a.round_calls:
                raise RuntimeError('Run budget exhausted')
            self.call_count += 1
            identifier = f'{self.call_count:04d}-{agent}'
        sent = self.compact(messages, agent)
        body = {'model':self.a.model, 'messages':sent, 'temperature':0, 'max_tokens':self.a.max_tokens,
                'reasoning_effort':'low', 'chat_template_kwargs':{'reasoning_strength':'low'},
                'stream':True, 'stream_options':{'include_usage':True}}
        self.event('request', id=identifier, agent=agent, prompt_chars=sum(len(x['content']) for x in sent))
        start = time.monotonic()
        first = None
        content = reasoning = ''
        usage = {}
        finish = error = None
        saw_done = False
        try:
            req = urllib.request.Request(self.a.base.rstrip('/')+'/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=min(self.a.request_timeout, max(.1,self.remaining_seconds()))) as response:
                with self.lock:self.active_responses.add(response)
                for line in response:
                    if self.cancelled.is_set() or self.remaining_seconds() <= 0:
                        raise TimeoutError('Cancelled or round/run wall limit reached')
                    if not line.startswith(b'data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == b'[DONE]':
                        saw_done = True
                        break
                    d = json.loads(raw)
                    if d.get('error'):
                        raise ValueError(str(d['error']))
                    if d.get('usage'):
                        usage = d['usage']
                    for c in d.get('choices', []):
                        delta = c.get('delta', {})
                        text, thought = delta.get('content') or '', delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if first is None and (text or thought):
                            first = time.monotonic()-start
                        content += text
                        reasoning += thought
                        finish = c.get('finish_reason') or finish
                if not saw_done or finish is None or 'completion_tokens' not in usage:
                    raise ValueError('Incomplete stream or missing token usage')
        except Exception as exc:
            error = str(exc)
        finally:
            with self.lock:
                if 'response' in locals():self.active_responses.discard(response)
        record = {'id':identifier,'agent':agent,'round':self.current_round,'phase':self.phase,'seconds':time.monotonic()-start,'ttft_seconds':first,
                  'content':content,'reasoning':reasoning,'usage':usage,'finish_reason':finish,'error':error}
        with self.lock:
            self.calls.append(record)
            self.tokens += usage.get('completion_tokens', 0)
            with (self.out/'calls.jsonl').open('a') as out:
                out.write(json.dumps(record,ensure_ascii=False)+'\n')
            write_json(self.out/'responses'/f'{identifier}.json', {'request':body,**record})
        self.event('response', **{k:v for k,v in record.items() if k not in {'content','reasoning'}})
        if error:
            raise RuntimeError(error)
        return content

    def check(self):
        with self.lock:
            self.check_count += 1
            target = self.out/'checks'/f'{self.check_count:03d}'
            target.mkdir(parents=True)
        command = [x.format(task_dir=self.task_dir,workspace=self.workspace,output=target,round=self.current_round or 1) for x in self.spec['validation_command']]
        start = time.monotonic()
        remaining=self.remaining_seconds()
        try:
            if self.cancelled.is_set() or remaining<=0:raise TimeoutError('Round/run budget exhausted before checker')
            p = subprocess.run(command, cwd=self.workspace, capture_output=True, text=True, timeout=min(120,remaining))
            try:
                detail = json.loads(p.stdout)
            except ValueError:
                detail = {'stdout':p.stdout[-10000:],'stderr':p.stderr[-10000:]}
                result={'passed':False,'infrastructure_error':True,'returncode':p.returncode,
                        'details':detail,'seconds':time.monotonic()-start}
                write_json(target/'result.json',result)
                self.event('check',result=result)
                return result
            passed = p.returncode == 0 and detail.get('passed',detail.get('success',True)) is not False
            changed = [name for name,sha in self.protected.items()
                       if not (self.workspace/name).is_file() or hashlib.sha256((self.workspace/name).read_bytes()).hexdigest()!=sha]
            if changed:
                passed = False
                detail['protected_files_changed'] = changed
            result = {'passed':passed,'returncode':p.returncode,'details':detail,'seconds':time.monotonic()-start}
        except (subprocess.TimeoutExpired,TimeoutError) as error:
            exhausted=self.cancelled.is_set() or self.remaining_seconds()<=0
            result={'passed':False,'infrastructure_error':not exhausted,'budget_exhausted':exhausted,
                    'error':str(error),'seconds':time.monotonic()-start}
        except Exception as error:
            result = {'passed':False,'infrastructure_error':True,'error':str(error),'seconds':time.monotonic()-start}
        write_json(target/'result.json', result)
        if result['passed']:
            self.round_first_pass.setdefault(self.current_round,time.monotonic()-self.started)
            if self.current_round == self.rounds[-1]['id'] and self.first_pass_seconds is None:
                self.first_pass_seconds = time.monotonic()-self.started
        self.event('check', result=result)
        return result

    def agent(self, name, owned, brief, limit, can_check=False):
        if name not in self.histories:
            self.histories[name] = [{'role':'system','content':PROTOCOL}, {'role':'user','content':self.shared+
                    f'\n\nYOUR ROLE: {name}\nEDITABLE FILES: {json.dumps(sorted(owned))}\n{brief}'}]
        messages = self.histories[name]
        current = next(r for r in self.rounds if r['id']==self.current_round)
        delta = '\n'.join(current.get('worker_instructions',{}).get(path,'') for path in sorted(owned))
        messages.append({'role':'user','content':f"CURRENT ROUND {self.current_round}: {self.round_title}\n"+
                         current.get('instructions','')+'\n'+delta+'\n'+brief+
                         '\nKeep previous requirements. Current workspace is authoritative; read files before revising stale content.'})
        status = 'call_limit'
        invalid_streak = 0
        turn = -1
        for turn in range(limit):
            try:
                content = self.request(name, messages)
            except Exception as error:
                self.event('agent_error', agent=name, error=str(error))
                status = 'request_error'
                break
            messages.append({'role':'assistant','content':content})
            try:
                if next(c for c in reversed(self.calls) if c['agent']==name)['finish_reason']=='length':raise ValueError('Response hit token cap; incomplete action is not executed')
                action = parse_action(content)
                kind = action.get('action')
                if kind == 'done':
                    if can_check:
                        result = self.check()
                        if result.get('infrastructure_error'):raise InfrastructureError('Protected checker failed to execute/return JSON')
                        if result['passed']:
                            status = 'done'
                            break
                        feedback = {'error':'Acceptance checks still fail. Continue repairing.','check':result}
                    else:
                        status = 'done'
                        break
                elif kind == 'read':
                    path = self.path(action.get('path'))
                    if not path.is_file() or path.stat().st_size > 40000:
                        raise ValueError('Read requires an existing small project file')
                    feedback = {'path':action['path'],'content':path.read_text()}
                elif kind == 'write':
                    filename = action.get('path')
                    if filename not in owned or filename not in self.editable:
                        raise ValueError('File is outside your editable ownership')
                    text = action.get('content')
                    if not isinstance(text,str) or len(text)>40000:
                        raise ValueError('content must be a complete UTF-8 source string under40000characters')
                    path = self.path(filename)
                    with self.lock:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        temporary = path.with_name(path.name+'.agent-tmp')
                        temporary.write_text(text)
                        temporary.replace(path)
                    self.event('file_write',agent=name,path=filename,bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())
                    feedback = {'written':filename,'bytes':len(text.encode())}
                elif kind == 'check' and can_check:
                    feedback = self.check()
                    if feedback.get('infrastructure_error'):raise InfrastructureError('Protected checker failed to execute/return JSON')
                    if feedback['passed']:
                        status = 'done'
                        break
                else:
                    raise ValueError('Unknown/unavailable action. Use read, write, done; integration may check.')
                invalid_streak = 0
            except InfrastructureError:
                raise
            except Exception as error:
                invalid_streak += 1
                with self.lock:self.tool_errors += 1
                feedback = {'error':str(error),'instruction':'Return a complete valid JSON tool action; preserve task behavior.'}
                self.event('tool_error',agent=name,error=str(error))
            messages.append({'role':'user','content':json.dumps({'tool_result':feedback},ensure_ascii=False)})
            if invalid_streak >= 3:
                status='three_invalid_actions'
                self.event('agent_stopped',agent=name,reason=status)
                break
        result = {'agent':name,'status':status,'turns':turn+1 if limit else 0}
        self.event('agent_finished',**result)
        return result

    def run(self):
        error = None
        infrastructure_issue = False
        watcher_done=threading.Event()
        watcher=threading.Thread(target=self.deadline_watch,args=(watcher_done,),daemon=True)
        watcher.start()
        try:
            if self.a.capture:
                capture_args=[]
                if self.spec.get('renderer'):capture_args += ['--renderer',str(self.task_dir/self.spec['renderer'])]
                capture_spec=self.spec.get('capture',{})
                if capture_spec.get('pages'):capture_args += ['--pages',','.join(capture_spec['pages'])]
                if capture_spec.get('entry'):capture_args += ['--entry',capture_spec['entry']]
                self.capture = subprocess.Popen(['node',str(HERE/'visual/record.mjs'),'--workspace',str(self.workspace),
                    '--output',str(self.out/'capture'),'--events',str(self.out/'events.jsonl'),
                    '--stop-file',str(self.out/'capture.stop'),'--start-time-unix',str(self.start_epoch)]+capture_args,
                    stdout=(self.out/'capture.log').open('w'),stderr=subprocess.STDOUT)
                until=time.monotonic()+45
                ready=self.out/'capture/capture-ready.json'
                while True:
                    try:initialized=ready.exists() and json.loads(ready.read_text()).get('frames',0)>0
                    except (OSError,ValueError):initialized=False
                    if self.cancelled.is_set():raise RuntimeError('Cancelled during recorder startup')
                    if initialized:break
                    if self.capture.poll() is not None or time.monotonic()>until:
                        raise InfrastructureError('Recorder failed initial-frame handshake')
                    time.sleep(.1)
            final={'passed':False}
            for current in self.rounds:
                if self.cancelled.is_set() or time.monotonic()-self.started >= self.a.wall_limit:break
                self.current_round=current['id'];self.round_title=current.get('title',str(self.current_round))
                self.round_started=time.monotonic();self.round_call_start=self.call_count
                token_start=self.tokens
                self.event('round_start',title=self.round_title,started_unix=time.time(),
                           instructions_sha256=hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest())
                results=[]
                if self.a.agents == 1:
                    self.phase='single';self.event('phase')
                    results.append(self.agent('single',self.editable,
                        'Complete the ENTIRE project, all editable files; check and repair.\n'+
                        json.dumps(self.spec.get('workers',self.spec.get('worker_assignments',[]))),
                        min(self.a.round_calls,self.a.max_calls-self.call_count),True))
                else:
                    workers=self.spec.get('workers',self.spec.get('worker_assignments'))
                    if len(workers)!=8:raise ValueError('Task must define exactly 8 disjoint workers')
                    self.phase='workers';self.event('phase')
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        futures=[pool.submit(self.agent,w['name'],{w.get('file',w.get('path'))},w['brief'],self.a.worker_calls,False) for w in workers]
                        results=sorted((f.result() for f in as_completed(futures)),key=lambda item:item['agent'])
                    initial=self.check()
                    if initial.get('infrastructure_error'):raise InfrastructureError('Protected checker failed to execute/return JSON')
                    if not initial['passed'] and not self.cancelled.is_set() and self.remaining_seconds()>0:
                        self.phase='integration';self.event('phase')
                        remaining=min(self.a.integration_calls,self.a.max_calls-self.call_count,
                                      self.a.round_calls-(self.call_count-self.round_call_start))
                        results.append(self.agent('integrator',self.editable,
                            'Inspect and repair actual current files; initial snapshot is stale, read before changing.\n'+
                            json.dumps({'workers':results,'failing_check':initial},ensure_ascii=False),max(0,remaining),True))
                final=self.check()
                if final.get('infrastructure_error'):raise InfrastructureError('Protected checker failed to execute/return JSON')
                self.worker_results.extend(dict(r,round=self.current_round) for r in results)
                row={'round':self.current_round,'title':self.round_title,'passed':final['passed'],
                     'wall_seconds':time.monotonic()-self.round_started,'calls':self.call_count-self.round_call_start,
                     'completion_tokens':self.tokens-token_start,'check':final,
                     'time_to_first_passing_check_seconds':self.round_first_pass.get(self.current_round),'workers':results}
                self.round_results.append(row);self.event('round_complete',**row)
                write_json(self.out/'rounds.json',self.round_results)
                write_json(self.out/'histories.json',self.histories)
        except Exception as exc:
            error=str(exc)
            infrastructure_issue=isinstance(exc,InfrastructureError)
            self.event('run_error',error=error)
            final={'passed':False,'error':error,'infrastructure_error':infrastructure_issue}
        finally:
            watcher_done.set();watcher.join(timeout=1)
            write_json(self.out/'histories.json',self.histories)
            (self.out/'capture.stop').write_text('stop\n')
            if self.capture:
                try:self.capture.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.capture.terminate();self.capture.wait(timeout=10)
        elapsed=time.monotonic()-self.started
        good=[c for c in self.calls if not c['error']]
        ttfts=[c['ttft_seconds'] for c in good if c['ttft_seconds'] is not None]
        summary={'task':self.spec.get('name',self.task_dir.name),'agents':self.a.agents,'model':self.a.model,
                 'started_unix':self.start_epoch,'finished_unix':time.time(),
                 'passed':final['passed'] and len(self.round_results)==len(self.rounds) and not self.cancelled.is_set(),'error':error,'wall_seconds':elapsed,'time_to_first_passing_check_seconds':self.first_pass_seconds,
                 'infrastructure_error':infrastructure_issue or bool(final.get('infrastructure_error')),
                 'completion_tokens':sum(c['usage'].get('completion_tokens',0) for c in self.calls),
                 'prompt_tokens':sum(c['usage'].get('prompt_tokens',0) for c in good),'calls':self.call_count,
                 'failed_calls':sum(bool(c['error']) for c in self.calls),'tool_errors':self.tool_errors,
                 'length_capped_calls':sum(c['finish_reason']=='length' for c in self.calls),
                 'ttft_median_seconds':statistics.median(ttfts) if ttfts else None,'workers':self.worker_results,
                 'rounds':self.round_results,'all_rounds_passed':len(self.round_results)==len(self.rounds) and all(r['passed'] for r in self.round_results),
                 'cancelled':self.cancelled.is_set(),'checks':self.check_count,'final_check':final,'capture_exitcode':self.capture.returncode if self.capture else None,
                 'note':'Actual generated actions execute; final cumulative correctness gates success. TPS includes reported completion tokens including reasoning; not useful-code throughput. Token admission uses completed usage and may overshoot with in-flight calls.'}
        summary['completion_tokens_per_wall_second']=summary['completion_tokens']/elapsed
        write_json(self.out/'summary.json',summary)
        self.event('finish',**summary)
        print(json.dumps(summary,indent=2))
        return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',default='http://127.0.0.1:8090/v1')
    p.add_argument('--model',default='glimmer-mlx4')
    p.add_argument('--agents',type=int,choices=[1,8],required=True)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--max-calls',type=int)
    p.add_argument('--worker-calls',type=int)
    p.add_argument('--integration-calls',type=int)
    p.add_argument('--round-calls',type=int)
    p.add_argument('--round-wall-limit',type=float)
    p.add_argument('--token-budget',type=int)
    p.add_argument('--max-prompt-chars',type=int,default=50000)
    p.add_argument('--request-timeout',type=float,default=600)
    p.add_argument('--wall-limit',type=float)
    p.add_argument('--capture',action='store_true')
    a=p.parse_args()
    bench=Benchmark(a)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda number,frame:bench.cancel(f'Signal {number}'))
    result=bench.run()
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
