# Lantern Commons Arts Weekend — project brief

We need an eight-page marketing site for a fictional community arts festival,
**Lantern Commons Arts Weekend**, on **12–13 June 2027** at **Alder Park,
Brookfield**. It should feel like a real small festival: warm, energetic and
welcoming, with strong typography, generous spacing and clearly presented useful
information. The current pages are plain placeholders. Read `content.json` for
our actual program, artists, venues and pricing; do not invent different facts.

Each editable `pages/NAME.html` is an HTML fragment inside the protected page
shell's `<main>`. Keep exactly one root `article id="page-NAME"`. Do not supply
`html`, `head` or `body`. The shell provides the eight-page navigation, viewport,
base reset and a disclosure that this is a fictional demo. Its renderer creates
`index.html` for home and `NAME.html` for the other seven pages. Link directly to
those local filenames. You own only your assigned file in a multi-agent run.
A single agent owns all eight; requirements are otherwise the same.

## Shared design direction

Use a consistent editorial direction: warm paper **#f5ebd9**, deep ink **#182e30**,
and violet brand accent **#7048c6**, with golden or soft orange supporting accents.
System fonts are sufficient; a serif display face paired with a clean sans-serif
body is welcome. Every page must include meaningful custom CSS in an inline
`style` block, scoped to its page article. Avoid affecting protected navigation.
Use clear sections, cards, whitespace and a convincing page composition rather
than a long unstyled text list. Keep source concise; no libraries or external
assets, remote fonts, scripts, inline event handlers or frames.

A supplied local abstract lantern illustration is available as
`assets/lantern.svg` in the rendered site. You may reference it with an informative
`alt`, make your own inline SVG, or use CSS shapes. It is an equal source asset for
all runs, not a completed page design. Do not depend on external image URLs.

Every page needs one descriptive `h1` (at least 12 characters), visible at 390px
and 1440px, with type at least 28px mobile / 34px desktop. Provide exactly one
primary action carrying `data-primary-cta`; give it a descriptive visible label,
a violet background **#7048c6**, and a target at least 40px high. The primary action
must actually be visible, not covered by opaque decorative layers (including
pointer-events-none pseudo-elements). The evaluator checks rendered pixels.
All page links must resolve; use `mailto:` only for the supplied demo contacts.
There must be no horizontal overflow. Forms need visible labels, correct input
types and live status regions. Do not claim to take payments or send real email.

## Page responsibilities and functional contracts

- **Home** (`pages/home.html`): introduce the full festival name, dates, park and
  city, summarize music/making/community, and provide an inviting hero. Primary
  action links to `tickets.html`; also link to `schedule.html`.
- **Lineup**: exactly three artist cards, each carrying `data-artist`, with the
  three names and disciplines in `content.json`. Primary action links to schedule.
- **Schedule**: show all six supplied events, each carrying `data-event` and
  `data-category="music"`, `"workshops"` or `"family"` as appropriate. Each event
  needs its title, `time datetime="DATE_TIME"` starting with the supplied ISO day,
  and its venue. Provide a visible `label for="schedule-filter"` and
  `select id="schedule-filter"` with values `all`, `music`, `workshops`, `family`.
  Include `id="schedule-empty" hidden`. Protected JS filters event cards when
  this select changes; do not override `[hidden]`. Primary action links to tickets.
- **Venues**: three `data-venue` cards for River Stage, Workshop Yard and Story
  Grove, with supplied descriptions. Include the park's street address and a
  link to `accessibility.html`. Primary action also links to that access guide.
- **Tickets**: show Day pass **$18** and Weekend pass **$30** in two `data-pass`
  cards. Make clear this is interest registration and no payment is taken.
  Provide `form id="ticket-form"`, visible `label for="ticket-email"`,
  `input id="ticket-email" type="email" required`, and a `button type="submit"`
  carrying `data-primary-cta`. Include `id="ticket-status" role="status"
  aria-live="polite"`. Protected JS gives local confirmation.
- **Accessibility**: give useful labeled guidance covering **Step-free routes**,
  **Quiet space** and **Accessible toilets**, and the supplied access email.
  Primary action links to `venues.html`.
- **Volunteer**: explain **Welcome crew** and **Workshop helpers** shifts. Provide
  `form id="volunteer-form"`, visible `label for="volunteer-email"`,
  `input id="volunteer-email" type="email" required`; visible
  `label for="volunteer-shift"`, and `select id="volunteer-shift"` with
  `welcome` and `workshops` options. A `button type="submit"` carries
  `data-primary-cta`; include `id="volunteer-status" role="status"
  aria-live="polite"`. Protected JS gives local confirmation.
- **FAQ**: four native `details` elements with descriptive `summary` questions
  covering families, rain, food and arrival. Say all ages are welcome, events
  continue in covered areas unless severe weather requires updates, picnics are
  welcome without glass, and arrival is via Riverpath entrance with bicycle
  parking near Workshop Yard. Primary action links to tickets.

The protected shell and JavaScript are not editable. The real browser checker
exercises navigation, visible content, schedule filtering, ticket/volunteer forms,
FAQ expansion, and responsive layout. CSS/DOM checks are functional acceptance,
not a complete aesthetic or accessibility certification. Final visual review is
separate. Finish actual file edits; saying the work is done does not pass a task.
