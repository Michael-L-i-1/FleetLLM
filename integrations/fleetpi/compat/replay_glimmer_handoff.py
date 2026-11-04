"""Model-free replay of installed Glimmer parsers and actual DelegatingParser.parse_delta.

Only parser AST is executed. Protocol/tokenizer stubs deliberately allow token IDs
to lead delivered text, reproducing detokenizer holdback without loading a model.
"""
import argparse
import ast
import dataclasses
import hashlib
import inspect
import json
import logging
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence

arguments = argparse.ArgumentParser(description=__doc__)
arguments.add_argument('--vllm-root', type=Path, required=True)
arguments.add_argument('--raw-response', type=Path)
options = arguments.parse_args()
ROOT = options.vllm_root.resolve()
logging.disable(logging.CRITICAL)

class Registry:
    parsers = {}
    @classmethod
    def register_module(cls, name):
        def register(parser):
            cls.parsers[name] = parser
            return parser
        return register

class Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items() if not exclude_none or v is not None}

class DeltaMessage(Record):
    def __init__(self, content=None, reasoning=None, tool_calls=None, **kwargs):
        super().__init__(content=content, reasoning=reasoning, tool_calls=tool_calls, **kwargs)

class Tokenizer:
    def decode(self, ids):
        return ''.join(map(chr, ids))

class Base:
    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer
    def adjust_initial_state_from_prompt(self, ids):
        pass

Base.engine_based_streaming = False

ENV = dict(
    re=re, json=json, Iterator=Iterator, Sequence=Sequence, Iterable=Iterable,
    ReasoningParser=Base, ToolParser=Base, DeltaMessage=DeltaMessage,
    ToolCall=Record, FunctionCall=Record, DeltaToolCall=Record, DeltaFunctionCall=Record,
    make_tool_call_id=lambda: 'fixture-call', init_logger=lambda _: logging.getLogger('fixture'),
    dataclass=dataclasses.dataclass, field=dataclasses.field,
    __name__='fixture', Path=Path, inspect=inspect, hashlib=hashlib,
    ReasoningParserManager=Registry,
)
module = types.ModuleType('fixture')
module.__dict__.update(ENV)
ENV = module.__dict__
sys.modules['fixture'] = module
module.__file__ = str(ROOT / 'reasoning/muse_glimmer_reasoning_parser.py')

def load(path, keep=None):
    source = ast.parse(path.read_text())
    source.body = [n for n in source.body if
        (isinstance(n, ast.ImportFrom) and n.module == '__future__') or
        (not isinstance(n, (ast.Import, ast.ImportFrom)) and (keep is None or getattr(n, 'name', None) in keep))]
    if not any(isinstance(n, ast.ImportFrom) and n.module == '__future__' for n in source.body):
        source.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    exec(compile(ast.fix_missing_locations(source), str(path), 'exec'), ENV)

load(ROOT / 'reasoning/muse_glimmer_reasoning_parser.py')
load(ROOT / 'tool_parsers/muse_glimmer_tool_parser.py')
load(Path(__file__).with_name('glimmer_handoff.py'))
abstract = ast.parse((ROOT / 'parser/abstract_parser.py').read_text())
delegating = next(n for n in abstract.body if isinstance(n, ast.ClassDef) and n.name == 'DelegatingParser')
methods = {'parse_delta', '_in_reasoning_phase', '_in_tool_call_phase', '_is_reasoning_end_streaming', '_extract_content_ids'}
source = ast.Module(body=[
    ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
    next(n for n in abstract.body if isinstance(n, ast.ClassDef) and n.name == 'StreamState'),
    ast.ClassDef(name='ActualDelegation', bases=[], keywords=[], body=[n for n in delegating.body if getattr(n, 'name', None) in methods], decorator_list=[]),
], type_ignores=[])
exec(compile(ast.fix_missing_locations(source), str(ROOT / 'parser/abstract_parser.py'), 'exec'), ENV)

class Harness(ENV['ActualDelegation']):
    def __init__(self, synchronized=False):
        parser_name = 'FleetPiGlimmerReasoningParser' if synchronized else 'MuseGlimmerReasoningParser'
        self._reasoning_parser = ENV[parser_name](Tokenizer())
        self._tool_parser = ENV['MuseGlimmerToolParser'](Tokenizer())
        self._stream_state = ENV['StreamState']()
        self._engine_based = False
    def _initialize_history_tool_call_cnt(self, request):
        pass
    def extract_reasoning_streaming(self, **kwargs):
        return self._reasoning_parser.extract_reasoning_streaming(**kwargs)
    def _extract_tool_calls_streaming(self, **kwargs):
        for key in ('tool_call_idx', 'tool_call_id_type', 'function_name_returned'):
            kwargs.pop(key)
        return self._tool_parser.extract_tool_calls_streaming(**kwargs), False
    def finalize_generation(self, delta, request, state):
        return delta
    def _flush_engine_parsers(self, delta):
        return delta

REQUEST = Record(include_reasoning=True, tool_choice='auto', tools=[Record(function=Record(name=name)) for name in ('ls', 'read')])
TEXT = ' to=self<|message|>Inspect workspace.<|eom|><|start|>assistant to=ls.ls<|message|><atem:function_calls><atem:invoke name="ls.ls"><atem:parameter name="path">/work</atem:parameter></atem:invoke></atem:function_calls><|eom|>'

def replay(text, chunks, lag=0, synchronized=False):
    parser = Harness(synchronized)
    visible = []
    calls = []
    previous_text_end = 0
    previous_id_end = 0
    transitions = []
    for id_end in chunks:
        final = id_end == len(text)
        text_end = len(text) if final else max(0, id_end - lag)
        was = parser._stream_state.reasoning_ended
        delta = parser.parse_delta(text[previous_text_end:text_end], list(map(ord, text[previous_id_end:id_end])), REQUEST, finished=final)
        if not was and parser._stream_state.reasoning_ended:
            transitions.append({'id_end': id_end, 'text_end': text_end, 'handoff_text': parser._stream_state.previous_text})
        if delta:
            if delta.content:
                visible.append(delta.content)
            calls.extend(delta.tool_calls or [])
        previous_text_end, previous_id_end = text_end, id_end
    return {'calls': len(calls), 'functions': [call.function for call in calls], 'content': ''.join(visible), 'transitions': transitions}

checks = 0
baseline = replay(TEXT, list(range(1, len(TEXT)+1)))
assert baseline['calls'] == 1 and baseline['content'] == ''
checks += 1

for lag in (1, 6, 16, 28, 64):
    original = replay(TEXT, list(range(1, len(TEXT)+1)), lag)
    patched = replay(TEXT, list(range(1, len(TEXT)+1)), lag, True)
    assert original['transitions'][0]['handoff_text'] == '', original
    assert patched['transitions'][0]['handoff_text'].startswith('to=ls.ls<|message|>'), patched
    assert patched['calls'] == 1 and patched['content'] == '', patched
    assert patched['functions'][0]['name'] == 'ls'
    assert json.loads(patched['functions'][0]['arguments']) == {'path': '/work'}
    checks += 1
    print(json.dumps({'case': 'delivered-text-lag', 'lag': lag,
                      'original_discarded_handoff': True, 'patched_calls': patched['calls']}))

for text in (TEXT, TEXT.replace('<|eom|>', '<|eot|>', 1), TEXT.replace('/work', '/work/日本語')):
    for split in range(1, len(text)):
        result = replay(text, [split, len(text)], synchronized=True)
        assert result['calls'] == 1 and not result['content'], (split, result)
        checks += 1

direct = TEXT[TEXT.index('to=ls.ls'):]
multiple = TEXT + '<|start|>assistant ' + direct.replace('ls.ls', 'read.read')
assert replay(direct, list(range(1, len(direct)+1)), 6, True)['calls'] == 1
assert replay(multiple, list(range(1, len(multiple)+1)), 6, True)['calls'] == 2
checks += 2
for recipient in ('self', 'user'):
    echoed = direct.replace('to=ls.ls', f'to={recipient}', 1)
    assert replay(echoed, list(range(1, len(echoed)+1)), 6, True)['calls'] == 0
    checks += 1

parser = ENV['FleetPiGlimmerReasoningParser'](Tokenizer())
assert not parser.is_reasoning_end(list(map(ord, 'system ATEM example <atem:function_calls><|start|>assistant')))
assert Registry.parsers['fleetpi_glimmer_handoff'] is ENV['FleetPiGlimmerReasoningParser']
checks += 2

with tempfile.TemporaryDirectory(prefix='fleetpi-parser-hash-') as directory:
    changed_root = Path(directory)
    for relative in ENV['EXPECTED_SOURCES']:
        target = changed_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    with (changed_root / 'parser/abstract_parser.py').open('a') as file:
        file.write('\n# unreviewed source change\n')
    try:
        ENV['verify_sources'](changed_root)
    except RuntimeError as error:
        assert 'source mismatch' in str(error)
    else:
        raise AssertionError('source hash guard accepted changed parser')
    checks += 1

# This separate pre-existing defect is deliberately not hidden by the predicate fix.
# Report it as a limitation, not as successful conformance of malformed framing.
missing_eom = TEXT.replace('<|eom|><|start|>assistant ', ' ')
whole = replay(missing_eom, [len(missing_eom)], synchronized=True)
incremental = replay(missing_eom, list(range(1, len(missing_eom)+1)), synchronized=True)
assert whole['calls'] == 0 and incremental['calls'] == 1
print(json.dumps({'known_unfixed': 'missing reasoning eom plus bare tool header',
                  'whole_chunk_calls': whole['calls'], 'incremental_calls': incremental['calls']}))
if options.raw_response is not None:
    raw = json.loads(options.raw_response.read_text())
    text = raw['choices'][0]['text']
    broken_opening = 'name="ls<|message|>'
    assert broken_opening in text, 'raw fixture is not the captured malformed first invoke'
    expected = text.split(broken_opening, 1)[1]
    for patched in (False, True):
        for chunking in ('whole', 'character'):
            chunks = [len(text)] if chunking == 'whole' else list(range(1, len(text)+1))
            result = replay(text, chunks, synchronized=patched)
            assert result['calls'] == 0 and result['content'] == expected, result
            checks += 1
            print(json.dumps({'raw_malformed_call': True, 'patched_predicate': patched,
                              'chunking': chunking, 'executable_calls': 0,
                              'parameter_tail_chars': len(expected)}))
print(json.dumps({'passed': checks, 'gpu_or_model_loaded': False,
                  'scope': 'source-hash-guarded parser AST; lightweight protocol/tokenizer stubs'}))
