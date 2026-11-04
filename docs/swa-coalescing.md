# Shared-prefix SWA coalescing (experimental; default off)

`--shared-prefix-coalescing` aliases existing `--recurrent-prefix-dedup`.
For an SWA target **without a learned drafter**, stop prompt prefill at the
nearest live waiting child's exact common-prefix endpoint. The existing
next-step resolver then forks full/SWA attention pages before the leader
advances. This preserves the page holding the earliest window token; previously
an overshooting prefill chunk could free that page before the waiting fork.

Example: shared prefix6000, window2048, budget128. A chunk ending6016 releases
SWA page247, needed by a fork at6000. The opt-in clamp ends at6000 instead.
No generation-phase checkpoints or changed attention arithmetic are introduced.
Recurrent behavior is unchanged. Learned DFlash drafting is explicitly excluded
from this SWA change; copying pending teacher captures into child draft context
has not been validated. N-gram drafting remains allowed. Existing attention
waiting/forking defaults remain unchanged when this option is absent.

Fixture uses the actual engine_create/engine_step scheduler with a tiny exact
full/SWA fake backend, not a reimplementation of scheduling. It checks6000-token
fanout off0/on42000 shared tokens; aligned, unaligned and staggered lengths;
window-page presence; deterministic output and retained KV content; full/SWA tail
copy-on-write and refcounts; released/omitted leader fallback; and constrained
pool operation. This proves scheduler/page ownership properties, not real-model
GPU output parity or performance. Real GPU prefill chunking can change floating
point reductions, so generated-token differences must be assessed separately.

CPU sanitizer commands (after root CPU handoff):

```sh
cd /tmp/fleet-glimmer-coalesce
make NO_METAL=1 BUILD=/tmp/fleet-swa-coalesce-asan \
  CFLAGS='-O1 -g -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -mcpu=native -D_GNU_SOURCE -D_DARWIN_C_SOURCE -fsanitize=address,undefined' \
  -j2 /tmp/fleet-swa-coalesce-asan/libfleetllm.a
cc -O1 -g -fsanitize=address,undefined -Isrc scripts/swacoalescecheck.c \
  /tmp/fleet-swa-coalesce-asan/libfleetllm.a -lpthread -lm \
  -o /tmp/fleet-swacoalescecheck-asan
ASAN_OPTIONS=detect_leaks=0 /tmp/fleet-swacoalescecheck-asan
```

Root owns real-model GPU tests and native performance builds. Do not combine
this candidate with the earlier experimental generation-checkpoint branch.
Adopt only after sanitizer checks and a real coordinated workload comparison
show acceptable quality and throughput, using identical target/quantization,
concurrency, input histories, model-free speculation flags, and memory budgets.

CPU validation passed with ASAN+UBSAN across the whole linked library. macOS
AddressSanitizer rejects `detect_leaks=1`; leak detection is therefore disabled,
while explicit tests require every full/SWA page to return to its free pool and
free the backend-owned pool allocations. Logs: `/tmp/fleet-swacoalesce-asan.log`
and `/tmp/fleet-swa-coalesce-build.log`. Unaligned6000-token eight-way fanout
reduced actual fake-backend prefill rows48128 ->6128 with unchanged outputs and
retained KV. Aligned6016 already coalesced in the control; both used6144 rows.
These counts describe fixture scheduling work, not measured GPU throughput.

Root's matched real-GPU eight-request JSON file-edit pilot passed. With identical
binary, Muse Glimmer MLX four-bit model, prompts, output caps, and n-gram k15:

| Fleet setting | Wall seconds | Generated tokens | Generated tok/s | Shared prompt tokens | Forward rows |
|---|---:|---:|---:|---:|---:|
| Coalescing off | 136.5176 | 7815 | 57.245 | 0 | 64051 |
| Coalescing on | 44.9502 | 7815 | 173.859 | 46445 | 17283 |

All eight complete response messages (reasoning and content) were identical
byte-for-byte between arms. Each response was a valid JSON write action and
preserved the requested HTML edit. This is approximately **3.04x Fleet versus
Fleet on this shared-prefix pilot**, not a competitor comparison or a completed
multi-turn coding-task claim. Root owns the measurement and preserved the
unchanged native GPU binary hash. The planned main-worktree result provenance
is `results/glimmer-festival/setup/json-edit-pilot-summary.json`; this isolated
branch does not duplicate or modify those benchmark artifacts.
