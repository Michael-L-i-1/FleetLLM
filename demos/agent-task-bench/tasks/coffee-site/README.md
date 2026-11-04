# Common Ground website task

A fictional nonprofit cafe's intentionally plain site must become a cohesive,
responsive marketing site. Eight editable HTML fragments support eight independent
workers; a single-agent arm owns those same eight files. Each fragment may contain
scoped inline CSS and SVG. Shared behavior, assembly and validation are protected
outside the editable workspace. The task brief gives all required content and
selectors; no hidden business requirements or private expected design are used.

The initial fixture deliberately fails meaningful requirements: missing navigation,
mission disclosure, drink cards, impact figures, events, visit details, input labels,
custom styling and mobile layout. Search and newsletter behavior are implemented
in protected JavaScript so the task focuses on actual markup/design integration.
Models must provide the correct accessible markup for those behaviors to work.

From the repository root:

```sh
python3 demos/agent-task-bench/tasks/coffee-site/render.py \
  --workspace demos/agent-task-bench/tasks/coffee-site/fixture \
  --output /tmp/coffee-before
python3 demos/agent-task-bench/tasks/coffee-site/check.py \
  --workspace demos/agent-task-bench/tasks/coffee-site/fixture \
  --output /tmp/coffee-before-check
```

`render.py` returns JSON with `index` and `sha256`; it atomically replaces the
standalone output `index.html`. A recorder can re-render after edits and load it
with `file://` without a server. Output should be outside the model's editable
files. CSP allows the protected app script, inline styling and data images only.

`check.py` returns JSON and exits 0 only if all checks pass. It stores `results.json`,
the assembled site under `site/`, and full-page `390.png` / `1440.png` screenshots.
Its browser runner uses the already pinned Playwright dependency under
`demos/site-refactor/node_modules`; no new dependency or external network is needed.
The task manifest's validation/render commands use `{task_dir}`, `{workspace}`,
and `{output}` placeholders for the common benchmark harness.

Checks cover 57 independent assertions across both viewports: semantic headings,
visible navigation, real internal link targets, supplied content, menu search and
its empty state, valid event dates, accessible newsletter signup, no horizontal
overflow, no model scripts/frames, no remote assets, and no browser errors. The
styling check only establishes that CSS was supplied. Screenshots need separate
visual review; these tests do not certify beauty, WCAG compliance or conversion.

Fixture feasibility was checked without inference. The unchanged starter failed
29 of 57 checks. A temporary deliberately minimal implementation passed all 57,
including real search and form interactions; that synthetic implementation is not
part of the fixture, is not a model result, and is not used in benchmark runs.
`self-check.json` preserves those check summaries. `baseline-hashes.json` identifies
starter fragment contents; it is diagnostic, not an aesthetic grading mechanism.
