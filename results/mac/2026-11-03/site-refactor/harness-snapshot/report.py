#!/usr/bin/env python3
"""Build the saved experiment report after comparisons finish."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
args = parser.parse_args()
r = args.results
rows = {x['run']: x for x in json.loads((r / 'analysis.json').read_text())}
main = ['fleet-mlx4-span-01', 'fleet-mlx4-span-02', 'vllm-mlx4-standard-01', 'llama-q4km-standard-01']
lines = ['# Four-worker website refactoring on M5 Pro', '',
         'One coordinator delegates four isolated file-owning workers, integrates their edits, and runs browser checks. '
         'The primary comparison follows the user’s requested policy: optimized FleetLLM against standard GPU serving in vLLM Metal and llama.cpp. '
         'Native defaults remain enabled; no optional speculation, prompt lookup or cache-reuse tuning is enabled in the comparators.', '',
         'All rows use **Muse Glimmer 30B**. “llama.cpp” names the inference engine, not a Llama model. Q4_K_M is the GGUF four-bit quantization.', '',
         '## Full task', '',
         '| Run | Quantization | Checks | Wall time | Generated tokens | Task-wide tok/s | Calls | Tool/JSON errors | Median first-output latency |',
         '|---|---|---|---:|---:|---:|---:|---:|---:|']
for name in main:
    if name not in rows:
        continue
    x = rows[name]
    minutes = int(x['seconds']//60)
    seconds = x['seconds']-60*minutes
    quant = 'GGUF Q4_K_M' if name.startswith('llama') else 'MLX affine 4-bit / group 64'
    display = {'fleet-mlx4-span-01': 'Fleet optimized · first trial', 'fleet-mlx4-span-02': 'Fleet optimized · final repeat', 'vllm-mlx4-standard-01': 'vLLM Metal · standard', 'llama-q4km-standard-01': 'llama.cpp · standard'}.get(name, name)
    lines.append(f"| {display} | {quant} | {'pass' if x['success'] else 'FAIL'} | {minutes}:{seconds:04.1f} | {x['completion_tokens']:,} | {x['aggregate_completion_tokens_per_second']:.2f} | {x['calls']} | {x['tool_errors']} | {x['median_ttft_seconds']:.2f} s |")
lines += ['', 'Generated tokens include reasoning and control tokens as reported by the API. Task-wide rate includes model requests, tool execution, integration and final checks. '
          'Weights are loaded and an unrelated READY warmup completes before timing. Fixture preflight and server startup are excluded. '
          '“Checks” means the functional/browser gates, not a general code-quality or aesthetics score.', '',
          '## Same worker prompts, four concurrent requests', '',
          'Each engine receives the same four real worker histories, selected from the first Fleet run, with a 256-token cap per request. '
          'This measures bounded inference, not completed coding tasks. Early stops and different generated tokens can change the active batch mix.', '',
          '| Run | Prompt tokens | Generated tokens | Wall time | Aggregate tok/s |',
          '|---|---:|---:|---:|---:|']
for name in ['replay-fleet-mlx4-span-01', 'replay-fleet-mlx4-ngram-01', 'replay-vllm-mlx4-standard-01', 'replay-llama-q4km-standard-01', 'replay-fleet-q4km-ngram-01']:
    if name in rows:
        x = rows[name]
        lines.append(f"| {name} | {x['prompt_tokens']} | {x['completion_tokens']} | {x['seconds']:.2f} s | {x['aggregate_completion_tokens_per_second']:.2f} |")
lines += ['', 'The same-GGUF rows control weight format between Fleet and llama.cpp. The MLX/GGUF rows use different quantization schemes; '
          'do not describe their difference as an engine-only speedup. Fleet’s GGUF Metal path uses different kernels from its MLX/NAX path.', '',
          '## Harness and optimization observations', '',
          'The llama.cpp trial failed while repairing broken search filtering: its next 16,569-token request exceeded the configured 16,384-token slot context. Its 14:15 wall time and token rate describe an unsuccessful attempt, not time to a working result. This is a limit of this trial configuration and generated repair trajectory, not proof that llama.cpp cannot complete the task. No completion-time speedup is claimed against this failed trial.', '']
for name in main:
    if name not in rows:
        continue
    x = rows[name]
    if 'worker_phase_seconds' in x:
        lines.append(f"- {name}: worker phase {x['worker_phase_seconds']:.2f} s, {x['worker_phase_completion_tokens_per_second']:.2f} aggregate tok/s; mean in-flight API requests over the full task {x['average_inflight_api_requests']:.2f}; {x.get('write_calls', 0)} file writes and {x.get('check_calls', 0)} model-requested checks, {x.get('failed_check_feedbacks', 0)} failed-check feedbacks.")
lines += ['', '- Fleet uses Metal/NAX, target-verified DFlash, prompt lookup, prefix sharing, continuous batching and admission. FP16 KV, 4 GB pool; no layer skipping or activation sparsity.',
          '- The span-reuse flag was supplied, but span processing is disabled while a DFlash drafter is attached (`src/engine.c`: `reuse_phase` is gated by `!e->draft`). Span reuse was therefore inactive in these DFlash runs; this was not a combined span-plus-DFlash test. The first span and no-span runs have identical assistant content/reasoning and final files; see fleet-span-vs-exact-equality.json. Do not attribute their timing difference to span reuse.',
          '- The first Fleet run reused 70,830 of 82,577 requested prompt tokens (85.8%, API counters). vLLM’s stream did not report comparable per-request cached-token counts; consult server counters rather than interpreting missing data as zero hits.',
          '- Coordinators can rewrite entire worker files during integration. This is real agent work and is included in timing; a shared design/selector contract and targeted integration patches are the next harness improvements.',
          '- Greedy sampling does not guarantee identical outputs across engines or quantizations. Compare call counts, generated tokens, screenshots and final code alongside completion time.', '',
          '## Reproducibility and limits', '',
          '- Hardware: Apple M5 Pro, 20-core GPU, 48 GB unified memory, AC power. Server commands, revisions, checkpoint SHA-256 and source hashes are saved alongside this report.',
          '- MLX target revision: `3e7677d7a40d348a3daba263a2b1c0aa41910710`; drafter revision: `c676cce91658f9568aaf264a416211e2782735b7`. GGUF provenance is in `gguf-source.json`.',
          '- The installed vLLM Metal could not serve Glimmer. An isolated copy backports [upstream Glimmer support](https://github.com/vllm-project/vllm-metal/pull/1120) and [the normal selective-logits path](https://github.com/vllm-project/vllm-metal/pull/1220). The exact adapted diff is `vllm-applied-compat.patch`; all 12 upstream Glimmer checks pass in `vllm-compat-overlay-tests.txt`. The system installation is unchanged.',
          '- Comparator baseline policy and exact argv are in `baseline-policy.json`, `vllm-standard.command.json`, and `llama-standard.command.json`. GPU memory/context caps configure the workload; GPU compute is not artificially restricted. vLLM’s native prefix-caching default remains enabled.',
          '- First Fleet runs had a background model download. Standard baselines ran after it finished. This is a small exploratory sample without confidence intervals, not a universal speedup claim.',
          '- Browser checks cover search, signup, labels, preserved script/IDs and horizontal overflow at 390/1440 px. Screenshots require visual review; no WCAG certification or broad model-accuracy evaluation was performed. Screenshot capture now resets scroll to avoid a sticky-header artifact after form interaction; acceptance gates are unchanged.',
          '- RSS/CPU process-tree samples are retained, but RSS is not a reliable measure of total Metal allocation and sums may double-count shared pages. GPU/KV allocations are recorded in startup logs.',
          '- Earlier tuned-vLLM n-gram experiments are supplementary and excluded from the primary baseline table. No llama.cpp speculative experiment was run after the user changed the comparison policy.', '',
          '## Saved evidence', '',
          'Per-run `*.summary.csv`, `*.summary.json`, `*.calls.json`, `*.events.jsonl`, server logs, exact config and resource samples are in this folder. '
          '`analysis.json` contains expanded latency and phase metrics. `artifacts/` archives retain run transcripts, code, patches and screenshots; no model weights are included.', '']
(r / 'README.md').write_text('\n'.join(lines))
print(r / 'README.md')
