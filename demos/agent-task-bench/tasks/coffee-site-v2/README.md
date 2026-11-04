# Common Ground website task — version 2

This is a new version of the eight-section coffee-site task. Version 1 remains
unchanged. The plain starter fragments, protected behavior and renderer are copied
from version 1; the brief clarifies that the hero headline and both hero calls to
action must actually be visible, including beneath decorative overlays.

The new evaluator adds six rendered-pixel checks: headline, menu CTA and mission
CTA at 390px and 1440px. DOM presence, CSS layout visibility and hit testing cannot
detect all occlusion: a fully opaque `::before` with `pointer-events: none` can
cover text while hit testing still returns the text underneath.

## Pixel probe

For each target, `visible-pixels.mjs` scrolls it into view, pauses animations and
temporarily disables transitions. It captures the same clipped rectangle twice
unchanged, once with the target subtree set to `visibility: hidden !important`,
and once after restoring every original inline `style` attribute exactly.
Visibility does not remove the element's layout box. Target descendants are also
hidden so an explicit child `visibility: visible` does not invalidate the probe.

PNG pixels are decoded with Playwright's bundled decoder, with no new dependency.
Pixels changing between unchanged/restored controls are excluded as unstable.
At least 24 stable pixels must change by at least 8 in an RGB channel when the
target is hidden. This is a conservative contribution test, not a prescribed
font, color, layout, contrast ratio or beauty score. It detects an entirely
occluded or transparent target; it does not certify full legibility, aesthetics,
WCAG compliance or every possible partial occlusion.

All temporary style changes, transition guards and animation states are restored.
Diagnostic before/hidden/restored PNG crops and per-target pixel counts are saved
in the validator output. Browser execution explicitly uses Chromium
`--disable-gpu`, so validation does not run a GPU workload beside model inference.

## Interface

From the repository root:

```sh
python3 demos/agent-task-bench/tasks/coffee-site-v2/render.py \
  --workspace demos/agent-task-bench/tasks/coffee-site-v2/fixture \
  --output /tmp/coffee-v2-before
python3 demos/agent-task-bench/tasks/coffee-site-v2/check.py \
  --workspace demos/agent-task-bench/tasks/coffee-site-v2/fixture \
  --output /tmp/coffee-v2-before-check
node demos/agent-task-bench/tasks/coffee-site-v2/probe-selfcheck.mjs
```

The renderer atomically writes a self-contained `index.html`, returning JSON
`index` and `sha256`. The validator returns JSON, exits 0 only when every check
passes, and saves `results.json`, full-page mobile/desktop screenshots, pixel-probe
crops and the rendered site. The task manifest uses `{task_dir}`, `{workspace}`
and `{output}` placeholders. Eight model-editable fragments remain independent;
the single-agent arm owns the same eight files. Protected validation remains
outside the model-editable workspace.

## Evaluator validation, without inference

`self-check.json` records the following 63-check runs:

- Unchanged starter: fails 33 checks, as intended.
- Temporary minimal feasibility implementation: passes all 63. This temporary
  implementation is not the benchmark fixture and is not a model result.
- Actual Fleet eight-agent version-1 artifact: passes all 63, including all six
  new visible-pixel checks.
- Actual vLLM eight-agent version-1 artifact: fails exactly the six new checks.
  Each required hero target contributes zero stable pixels at both viewports;
  the original 57 checks still pass. This is retrospective grader validation,
  not a version-2 model-run result.

The focused probe self-check additionally accepts ordinary visible text and a
visible nested child with explicit `visibility: visible !important`, and rejects
opaque pointer-events-none occlusion, transparent targets and animated opaque
occlusion. Every case verifies exact restoration of inline styles. No inference
or model/GPU job is part of these feasibility checks.

The marketing/content/search/form/responsiveness checks from version 1 remain.
They use the already pinned Playwright dependency in `demos/site-refactor` and
require no external assets or network. Separate visual review remains necessary.
