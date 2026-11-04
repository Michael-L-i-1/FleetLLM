# Lantern Commons: eight pages, three real editing rounds

A fictional community arts weekend needs an eight-page marketing website: home,
lineup, schedule, venues, tickets, accessibility, volunteer and FAQ. Each worker
owns one small HTML fragment. The single-agent arm owns the exact same eight
fragments and receives the same project and stakeholder requirements.

The starter is intentionally plain and incomplete. The shared `brief.md` and
`content.json` total about 8 KB of genuine project context: actual program facts,
page ownership, supplied local artwork, design direction, selectors and functional
contracts. There is no repeated padding or completed design in the starter.
Future stakeholder updates live in the protected task manifest and are delivered
as later user messages, not exposed in the initial editable workspace.

## Rounds

1. **Initial redesign:** custom page compositions, supplied artist/event/venue
   content, real local schedule filtering, ticket/volunteer interest forms and
   native expanding FAQ. The initial primary action brand is violet.
2. **Family stakeholder update:** new teal actions, a useful admission/hours/fees
   advisory on every page, and page-specific family highlights. The volunteer
   form gains a genuinely selectable family-welcome shift. Adult prices and all
   earlier functionality remain intact.
3. **Browser polish:** refine mobile/desktop behavior, give primary actions a
   distinct visible keyboard-focus treatment, and add a consistent access-help
   strip with functional guide/contact links. Earlier requirements are cumulative.

Models must issue real JSON write actions with page source in `content`; raw HTML
responses are not accepted as tool actions. Tests and shared behavior are outside
the model-editable workspace. No external network, image generation or remote
libraries are needed. The supplied abstract SVG is an equal project asset for all
arms; models still implement the actual pages and styles.

## Renderer and checker

Run from the repository root:

```sh
python3 demos/agent-task-bench/tasks/festival-site/render.py \
  --workspace demos/agent-task-bench/tasks/festival-site/fixture \
  --output /tmp/festival-before
python3 demos/agent-task-bench/tasks/festival-site/check.py \
  --workspace demos/agent-task-bench/tasks/festival-site/fixture \
  --output /tmp/festival-before-check --round 1
```

The renderer deterministically writes `index.html`, `lineup.html`, `schedule.html`,
`venues.html`, `tickets.html`, `accessibility.html`, `volunteer.html`, `faq.html`,
and the supplied local SVG. It atomically replaces only changed bytes. JSON stdout
contains `index`, `sha256`, `pages`, `source_to_output` and per-page hashes. There
are no generated timestamps or per-run nonces. The recorder can therefore show
actual changes and reuse unchanged page thumbnails without inventing site content.

The checker accepts `--round 1`, `2` or `3`, with cumulative requirements. It
returns JSON and exits 0 only if all checks pass, saving results and mobile/desktop
screenshots. Real Chromium interactions exercise the schedule select, both forms
and a native FAQ disclosure. Each page's required headline and action must
contribute visible pixels at both 390px and 1440px; the versioned coffee task's
validated before/hidden/restored PNG probe catches opaque pointer-events-none
occlusion that DOM presence/hit testing misses. Temporary inline styles and
animations are restored. Keyboard-focus checks use real Tab navigation rather
than assuming programmatic focus activates `:focus-visible`.

Chromium uses `--disable-gpu`. The dependency is the repository's pinned
Playwright installation in `demos/site-refactor/node_modules`; alternatively set
`FESTIVAL_PLAYWRIGHT_MODULE` to its `@playwright/test/index.mjs`. No new browser or
PNG dependency is installed by the checker.

These are functional and structural acceptance checks. They do not certify
beauty, WCAG compliance, contrast ratios or real-world conversion effectiveness;
final visual review remains separate. Local forms only show confirmation and
never take payment or transmit signups.

## Model-free feasibility

The unchanged starter fails the final cumulative checker. Independent temporary
minimal implementations pass every round: 187/187 checks for round 1, 219/219 for
round 2 and 251/251 for round 3, including real form/filter/FAQ interactions and
visible heading/action probes at both widths. Each complete browser evaluation
took about 8 seconds with GPU disabled. `self-check.json` records the assertions.
Temporary implementations were removed; they are not benchmark inputs or model
results. No model or GPU job was run while constructing this fixture.
