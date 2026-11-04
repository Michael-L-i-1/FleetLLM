# Actual development capture

The recorder photographs the rendered workspace as model edits land. It does not
replace the model's output with a designed mockup. Chromium is launched with
`--disable-gpu` so capture does not contend for Metal inference. External network
requests and file requests outside the preview directory are blocked.

From the repository root:

```sh
node demos/agent-task-bench/visual/record.mjs \
  --workspace /path/to/run/workspace --output /path/to/run/capture \
  --events /path/to/run/events.jsonl --stop-file /path/to/run/capture.stop \
  --start-time-unix 1790000000 --interval 10
```

The default renderer is `tasks/coffee-site/render.py`, invoked as
`python3 render.py --workspace WORKSPACE --output CAPTURE/preview`. Override it
with `--renderer` (and optionally `--python`) for another trusted fixture.
`--start-time-unix` is the benchmark start epoch in seconds; include capture
initialization in the timed span if that is where the shared clock begins.
Wait for `capture-ready.json` with `frames > 0` before model requests to guarantee
an unedited baseline screenshot. Writes cause additional captures, throttled to
once per three seconds; `--min-change-seconds` changes that threshold.

Write the stop file in the harness's `finally` block. The recorder takes a final
snapshot and a full-page image, closes Chromium, and writes
`capture-finished.json`. SIGINT/SIGTERM also request a final capture. Each
`frames.jsonl` entry records its actual elapsed screenshot time, reason, path,
and the latest harness event if available. Render errors are preserved in the
manifest rather than being silently presented as a successful edit. Last good
frames remain usable. `capture.json` records dimensions and sampling settings.

## Common-time comparison video

After all three sequential runs finish, write a JSON configuration such as:

```json
{
  "title": "Muse Glimmer 30B four-bit · coffee website",
  "speed": 20,
  "sample_seconds": 10,
  "fps": 12,
  "lanes": [
    {"label": "Fleet · 8 agents", "capture_dir": "/run/fleet/capture", "duration_seconds": 120, "status": "passed", "metrics": "120s · actual tok/s and memory from summary"},
    {"label": "vLLM-Metal · 8 agents", "capture_dir": "/run/vllm8/capture", "duration_seconds": 150, "status": "passed", "metrics": "150s · actual tok/s and memory from summary"},
    {"label": "vLLM-Metal · 1 agent", "capture_dir": "/run/vllm1/capture", "duration_seconds": 240, "status": "failed", "metrics": "240s · failed acceptance checks"}
  ]
}
```

These numbers are illustrative; populate them from actual run summaries. Then:

```sh
node demos/agent-task-bench/visual/compose.mjs comparison.json output-video-dir
```

Outputs are a 1920×800 H.264 `comparison.mp4`, `poster.png`, and an inspectable
`timeline.json`/`timeline.ffconcat` plus composite frames. All lanes use the same
elapsed-time axis and acceleration. The initial frame is shown before later
captures; a lane freezes at its final artifact only after its actual end time.
Passed, failed, timed-out and incomplete runs retain their real labels. If the
final render fails, the last successful snapshot is explicitly labeled; it is
not represented as the final rendered artifact. Final
metrics appear only after that lane ends. The last three video seconds hold all
final states. Sampling skips intermediate changes; this is a development
**timelapse**, not a continuous screen recording. A final snapshot may be captured
just after the harness's finish event; it shows the unchanged final artifact.

The compositor refuses missing screenshots or existing composite-frame folders.
Use fresh output folders. It requires the existing Playwright dependency under
`demos/site-refactor/node_modules` and `ffmpeg`; no new packages are installed.
