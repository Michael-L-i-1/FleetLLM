# Legacy Ledger: C to Rust migration

Replace the eight incomplete Rust modules with a correct, dependency-free Rust
port of the legacy C ledger CLI. `reference.c` is the read-only legacy source;
`main.rs` is the read-only Rust CLI and integration pipeline. Keep every public
function signature and the `Record` fields unchanged. Use Rust's standard
library only. Do not edit the reference, fixture data, main program, or validators.

This is one integrated application: CSV records become normalized, validated
transactions, duplicate IDs are removed, a date range is selected, and totals
are grouped by vendor. The same complete task is assigned to the single-agent
and multi-agent comparison arms.

| Editable file | Required behavior |
|---|---|
| `money.rs` | Parse strict signed decimal amounts into exact `i64` cents and format cents with two decimal places. Never use floating point. Accept optional `+`/`-`, at least one whole digit and optionally one or two fraction digits. Reject spaces, exponents, missing parts, excessive precision and overflow. Handle `i64::MIN` formatting. |
| `date.rs` | Validate exact ASCII `YYYY-MM-DD`, years 0001–9999, real Gregorian month lengths, and century-aware leap years. |
| `csv.rs` | Parse one CSV record. Preserve contents and empty fields. A field may start with a quote; doubled quotes escape one quote. A closing quote may be followed only by comma/end. Reject stray/unclosed quotes and any CR/LF. |
| `labels.rs` | Trim/collapse **ASCII whitespace** to ordinary spaces and lowercase **ASCII A–Z**. Preserve non-ASCII text unchanged; do not apply Unicode case folding or Unicode whitespace trimming. |
| `tags.rs` | Split on `;`, trim ASCII whitespace, discard empty segments, lowercase ASCII, reject any nonempty label containing characters other than ASCII letters/digits/`_`/`-`, and return sorted unique labels. |
| `filter.rs` | Return cloned records within an inclusive valid ISO date range, preserving input order. Invalid or reversed endpoints return an empty vector. |
| `summary.rs` | Group by the exact vendor string, count records, sum signed cents, retain zero totals, and sort lexicographically by vendor. Test totals do not overflow `i64`. |
| `audit.rs` | Keep the first record per exact, case-sensitive ID in input order. Return each repeated ID once, sorted lexicographically. Do not mutate inputs. |

ASCII whitespace means space, tab, CR, LF, vertical tab and form feed. Inputs
are valid UTF-8, ledger files contain single-line records, and acceptance data
is within the C program's fixed row/field/text limits. ISO dates sort in calendar
order only after validation. Money accepts leading zeroes but canonical output
never includes redundant zeroes or a plus sign.

Build the port with:

```sh
rustc --edition=2021 -C opt-level=1 main.rs -o /tmp/ledger
/tmp/ledger report fixtures/ledger.csv 2024-02-01 2024-03-31
```

The CLI also exposes `money VALUE`, `date VALUE`, `csv LINE`, `label TEXT`,
`tags TEXT`, `audit CSV_PATH`, `summary CSV_PATH`, and
`filter FROM TO CSV_PATH`. Invalid scalar inputs return `ERR`.

In a report, `accepted` counts all valid first-wins rows before range filtering;
`invalid` counts malformed rows, not duplicate IDs. `selected` preserves the
post-dedup input order. A summary includes only selected rows. The protected
acceptance checker tests each module API directly and compares three integrated
reports against both the C executable and an independent Python oracle,
including two seeded generated ledgers. Completion means **all 11 checks pass**.
