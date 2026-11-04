# Common Ground Coffee — real website redesign

Turn this plain local nonprofit cafe site into a polished, responsive marketing
website. Work in the existing eight section fragments; the protected renderer
assembles them into one page. Design a coherent editorial look: warm cream,
deep forest green, copper accents, generous spacing, large confident typography,
clear cards and buttons. Local system fonts, inline SVG and CSS shapes are fine.
Do not use external assets, JavaScript, remote fonts, scripts, frames or libraries.
The protected shell owns the viewport, base reset and behavior. Put a scoped
`<style>` block in each fragment and scope selectors to your section/header/footer.
Avoid changing styles for other sections. Every section needs meaningful custom
styling, not just altered copy. Preserve semantic HTML and accessible controls.

The brand is **Common Ground Coffee**, a fictional nonprofit in Palo Alto.
Mandatory copy is a testable contract; wording around it and design are yours:

1. **Header**: retain `id="top"`; brand link to `#top`; semantic `nav` with
   descriptive links to `#menu`, `#mission`, `#events`, `#visit`. Make navigation
   wrap or stack on narrow screens.
2. **Hero**: retain `id="hero"`; exactly one descriptive `h1` on the page,
   at least 12 characters; size at least 40px desktop / 28px mobile. Include
   contrasting calls to action linking to `#menu` and `#mission`. Convey
   excellent coffee funding opportunity; consider a large inline SVG/CSS visual.
3. **Mission**: `section id="mission"`, `h2`, explain explicitly that this is a
   **nonprofit** and **100%** of surplus funds **job training**. Keep it readable
   and emotionally credible; no invented claims of certification.
4. **Menu**: `section id="menu"`, `h2`, visible `label for="menu-search"`,
   `input id="menu-search" type="search"`. Exactly three drink cards bearing
   `data-menu-item` and appropriate `data-name`: **Espresso $3**, **Oat Latte $5**,
   **Cold Brew $4**. Keep these names and prices visible. Add
   `id="menu-empty" hidden` with a friendly no-results message. The protected
   script filters by `data-name`; do not hide content in your CSS using a rule
   that overrides the `hidden` attribute. All three cards must fit mobile width.
5. **Impact**: `section id="impact"`, `h2`; communicate these supplied annual
   figures: **2,500** community meals, **120** training hours, **35** graduates.
   Give each a meaningful label and use a legible responsive stats layout.
6. **Events**: `section id="events"`, `h2`; exactly three cards bearing
   `data-event`, each with an `h3`, a `time datetime="..."`, and a signup link
   to `#newsletter`: Community Table (2027-03-20), Open Mic (2027-04-03),
   Coffee Skills (2027-04-17). Include inviting short descriptions.
7. **Visit**: `section id="visit"`, `h2`, semantic `address` containing
   **42 Willow Street, Palo Alto**. Show Weekdays **07:00–18:00** and Weekends
   **08:00–16:00**; optional email `hello@commonground.example`. Do not embed a
   remote map. A local decorative map or location illustration is welcome.
8. **Newsletter/footer**: `section id="newsletter"`, `h2`,
   `form id="newsletter-form"`, visible `label for="email"`,
   `input id="email" type="email" required`, and a
   `button type="submit"`. Include `id="newsletter-status" role="status"
   aria-live="polite"`; protected JS writes success there. Include a short
   privacy reassurance, nonprofit identity, and useful internal footer links.

All IDs above are unique. Every internal anchor must resolve to a real ID.
At 390px and 1440px viewports the page must have no horizontal overflow and all
controls must work. Browser checks exercise menu search (including no results),
newsletter submission, headings, links, copy and responsiveness. Checks require
a meaningful inline style block in every module, but do not certify beauty,
contrast ratios, WCAG compliance or marketing effectiveness; final screenshots
also receive separate visual review.

Editable files are only `sections/01-header.html` through `08-footer.html`.
Workers own their assigned file. A single agent owns all eight and has the same
requirements. The protected assembler provides `<html>`, `<head>`, `<body>` and
`<main>`: fragments must not supply those outer documents. Do not edit protected
code. A task is complete only when the assembled site passes the checks, not
merely when an agent says it is finished.

## Visible content requirement

The hero headline and both hero calls to action must actually contribute visible
pixels at both viewports. A decorative absolute-positioned overlay must not cover
text or controls, even if it has `pointer-events: none`. Keep decorations behind
content (for example with an intentional stacking context and content layering).
The evaluator compares screenshots before and after temporarily hiding each
required target; DOM presence and hit-testing alone are not proof of visibility.
