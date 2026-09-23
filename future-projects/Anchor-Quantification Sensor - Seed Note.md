# Anchor-Quantification Sensor — Seed Note

A separate project, planned to run alongside Tell's 8-feature validated
panel and cross-reference against it, not as a 9th feature merged into
it. It will be built while completing the full live panel of higher-ed
institutions (15 scored so far, with 35 more planned, for 50 total).
That live panel is distinct from the 54-institution validated backtest
panel used for the classifier's own leave-one-out accuracy. This work is
a newly-identified project that turns Tell into something truly unique:
a stress detector for anchored systems, a measure of system health and
integrity post-anchoring, and a collapse tracker for unanchored
institutions, rather than simply the collapse tracker for liberal arts
institutions it began as. This work will be completed before the
4-adapter roadmap.

## Why separate from the panel

Tell's manuscript already establishes a two-layer distinction: the
classifier is the sensor, reading internal financial distress, and an
institution's anchor eligibility is the run-flat mechanism — external
support capable of preventing collapse without appearing anywhere in the
classifier's own math. So far that distinction has been qualitative:
real, and load-bearing, but not measured. This project is that
measurement.

It has to be built separately because there is no real anchor-failure/
closure data anywhere to fit a coefficient against — the same structural
data-scarcity problem already documented for `reserve_adequacy`'s
public-sector gap in the main tracker. Folding an unvalidated coefficient
into the panel's own fitted classifier would be exactly the kind of
unverified change this project doesn't make. The plan is to present the
two scores side by side — a second, independently computed number shown
alongside the classifier's, not merged into it.

## What it's for

One specific thing it's scoped to test: whether it can resolve a gap the
panel's own experimental `within_window_trend` feature couldn't. Sweet
Briar College's real, documented multi-year recovery after 2015 still
reads as high-entropy under that construction; Phoenix-Arizona comes back
flat. If anchor quantification separates these two the way their actual
outcomes diverge, that's real corroborating evidence for the approach —
and if it doesn't, that's a real negative result worth stating plainly.

## The two live test cases

- **Sweet Briar College (2015):** resolved by a real external
  philanthropic and legal rescue — a Virginia Attorney General-brokered
  settlement paired with a large alumnae-led fundraising campaign ("Saving
  Sweet Briar," widely reported at roughly $44 million raised in the years
  that followed). Capital and legal restructuring injected from outside
  the institution's own operating budget — the textbook shape of a RICD
  anchor.
- **University of Phoenix-Arizona:** no matching event. A planned sale to
  a University of Idaho-created nonprofit (announced 2023) was called off
  in June 2025 after the parties couldn't close, and its for-profit
  holding company moved toward an IPO instead (reported starting
  September 2025) — ownership restructuring within the same investor-owned
  structure, not an external anchor being added.

## Not yet built

What data actually feeds this: state appropriation/legal-intervention
records, ownership and acquisition filings, bond guarantees, donor/
rescue-commitment size relative to operating budget. What "cross-
referenced against the panel" looks like in practice is also not yet
finalized beyond the side-by-side principle above.

See the main tracker's [README — Known Gaps](../README.md#known-gaps--stated-honestly-not-smoothed-over)
for this in the context of the full 8-feature panel and its documentation
standard.
