# RICD Higher-Education Collapse Tracker

A Bayesian state-space model and validated classifier for institutional
financial collapse risk in U.S. higher education, built on the Recursive
Information-Container Dynamics (RICD) framework.

**The full RICD manuscript (the complete, domain-independent theory) lives at
[`docs/RICD 15.6 master.docx`](<docs/RICD 15.6 master.docx>)
(also available as [`.tex`](<docs/RICD 15.6 master.tex>)).**
Everything in this repository's code implements a real subset of that
framework (Parts 6, 7, 8, and 10.5b specifically) against U.S. higher-education
data; the manuscript itself is domain-independent and covers considerably more
than the tracker uses. The manuscript's own
[Integration Manifest](<reports/RICD Integration Manifest Complete.pdf>) records
every mechanism confirmed built into the framework, section by section, with
the exact manuscript text shown for each -- a standing verification record kept
specifically to be checked against the manuscript, not trusted on its own.

**The original source documents RICD was built from live in
[`source-documents/`](source-documents/):** the quartet of poems, the
Pentagonal Theorem of the Mathematical Nature of Evil, and Shaking Bowls —
Carmen Speer's own original creative and theoretical writing, which became
FDFM and RICS respectively, then nested into RICS-FDFM, then RICD, before
further revisions arrived at the manuscript above. An interlinear
**[Source Translation Ledger](<source-documents/Source Translation Ledger source poems explained mathematically.pdf>)**
sets RICD's own mathematics directly alongside the four original poems it
was built from, line by line, so the connection between the source material
and the formal framework is checkable rather than asserted. The folder's
`intermediate-development/` subfolder holds real, surviving milestones from
that path: an early FDFM application proposing a justice-system tracker
(directly cited by RICD 5.0's own editorial notes, not a lost document), the
expanded RICS-FDFM formalization, and earlier RICD versions (5.0
through 5.6). See that folder's own README for the full lineage.

**The narrative, findings, and process record of how this tracker was built
live in [`reports/`](reports/):** [`RICD Tracker Findings Final.pdf`](<reports/RICD Tracker Findings Final.pdf>),
[`RICD Tracker Narrative Final.pdf`](<reports/RICD Tracker Narrative Final.pdf>), and
[`RICD Tracker Process Log Final.pdf`](<reports/RICD Tracker Process Log Final.pdf>) --
the findings document, the narrative account of how each result was actually
reached, and the consolidated process log -- alongside
[`Higher Ed Sector Findings.pdf`](<reports/Higher Ed Sector Findings.pdf>) (a
separate analysis of what the results imply about the U.S. higher-education
sector as a whole), [`Claude's Account of Carmen's Role in Building RICD and the higher-ed tracker.pdf`](<reports/Claude's Account of Carmen's Role in Building RICD and the higher-ed tracker.pdf>)
and [`ChatGPT's Account of Its Own Role in the Early Development of FDFM, RICS, and RICD.pdf`](<reports/ChatGPT's Account of Its Own Role in the Early Development of FDFM, RICS, and RICD.pdf>)
(each AI collaborator's own account of working with Carmen on the framework and
the tracker, asked for and included so the actual division of labor is checkable
rather than asserted), [`RICD Adapter Instructional Manual.pdf`](<reports/RICD Adapter Instructional Manual.pdf>) (an instructional manual for
engineers working with the RICD adapter contract directly),
[`Actor Tracker Seed Note.pdf`](<reports/Actor Tracker Seed Note.pdf>) (a seed
note for a genuinely different kind of tracker planned for later), and the
complete [`RICD Integration Manifest Complete.pdf`](<reports/RICD Integration Manifest Complete.pdf>) described above.

The planned future tracker mentioned above is an
**actor tracker**, which would investigate the specific real decisions and
actors behind a collapse (board minutes, depositions, investigative
findings), rather than the aggregate financial and enrollment effects this
higher-ed tracker measures. The two are complementary, not competing —
see the seed note itself for why an actor tracker needs a different kind of
evidence entirely, and why it wasn't attempted here.

## What this is

A real, working pipeline — not a report about known outcomes — that:

1. Models an institution's official and operational representations as
   latent stochastic processes (`src/model.py`), including a genuine
   shock-type latent process for debt (`src/jump_diffusion.py`), since
   debt behaves as long stable stretches punctuated by rare large jumps
   rather than continuous drift.
2. Extracts eight validated, independently-tested features from real
   IPEDS data: informational trajectory (d_A trend and endpoint),
   resource debt severity and its own trend (δR, δR-trend), a debt-spike
   signature, a regime-classification fraction, scale-normalized reserve
   adequacy (endowment relative to enrollment, not to debt — see the
   note on that below), and a research-to-instruction expenditure ratio.
3. Applies a direct External Governance Attestation override for
   institutions with a real, independent, confirmed governance verdict
   (an accreditor's show-cause order or withdrawal), bypassing the
   statistical classifier entirely for those cases rather than fitting
   it as a weighted feature — confirmed empirically to be the correct
   approach (`src/classifier.py`).
4. Detects shared external shocks across the panel from cross-sectional
   co-movement alone (`src/common_cause.py`), run on raw adapter-level
   data rather than model output, since latent-variable smoothing
   washes out exactly the sharp signal this diagnostic needs.

## Validated result

100.00% leave-one-out cross-validated accuracy on a real, 54-institution
panel (23 confirmed closures spanning seven distinct collapse
mechanisms, 31 confirmed-stable comparisons), with zero misclassifications.
Reproduce this directly:

````bash
pip install -r requirements.txt
cd src
python -c "
from classifier import RICDClassifier, load_panel
panel = load_panel('../data/panel/panel.json')
clf = RICDClassifier()
acc, misclassified = clf.leave_one_out_accuracy(panel)
print(f'Accuracy: {acc:.2%}')
print(f'Misclassified: {misclassified}')
"
````

**A live public dashboard is at
[carmen-speer.github.io/Recursive-Information-Container-Dynamics](https://carmen-speer.github.io/Recursive-Information-Container-Dynamics/)**
(built from [`docs/index.html`](docs/index.html) via GitHub Pages), showing
both the validated 54-institution panel ([`docs/data/panel.json`](docs/data/panel.json))
and, in a separate section below it, real institutions scored live by the
pipeline described in Known Gaps below ([`docs/data/live_scores.json`](docs/data/live_scores.json)).

## The panel is a validation set, not a survey

The 54 institutions in `data/panel/panel.json` were assembled specifically
to test the classifier, including a deliberate rebalancing pass mid-project
after an audit found the panel had drifted toward roughly three times the
real-world rate of dramatic, easily-searchable closure cases. Nothing here
should be read as a claim about what fraction of U.S. higher education is
at risk. It supports a narrower, real claim: these specific mechanisms and
relationships showed up clearly enough in independently-verified data to
resolve a hard classification problem.

## Known gaps — stated honestly, not smoothed over

This is a live, ongoing project, not a finished product, and it's more
useful to state clearly what still needs real work than to imply
everything below is complete:

- **The live-scoring pipeline (`src/score_institution.py`) is confirmed
  working end to end against real, live data** (first successful run:
  2026-09-15, University of Michigan-Ann Arbor, UNITID 170976 -- a real
  classification produced from data fetched live from NCES and the
  College Scorecard API, not from any pre-downloaded local file). Its
  result is saved to [`docs/data/live_scores.json`](docs/data/live_scores.json)
  and shown in a "Live-scored institutions" section on the
  [public dashboard](https://carmen-speer.github.io/Recursive-Information-Container-Dynamics/).
  It still only scores one hardcoded institution per run, as a smoke
  test -- turning this into a tracker that scores a real, broader list
  of institutions on its own schedule is the next real step, genuine
  not-yet-done work, not a configuration problem.
- **IPEDS finance data has no stable API, and its distribution mechanism
  has changed more than once during this project -- now confirmed
  working, not just attempted.** `src/fetch_live_data.py` tries two
  real, live addresses in order: the newest year or two at
  `nces.ed.gov/ipeds/complete-data-files/<table>.zip`, falling back
  automatically to `nces.ed.gov/ipeds/datacenter/data/<table>.zip` for
  older years -- both confirmed directly against NCES's own live pages
  on 2026-09-15, via GitHub Actions (the only environment used on this
  project that can actually reach nces.ed.gov; every sandboxed AI
  environment used to develop this code is blocked from it). A live
  run that same day downloaded all 11 real fiscal years for a real
  institution and produced a real classification. NCES has changed
  this mechanism multiple times before across this project's history
  and may again -- automated re-scoring should still be monitored, not
  trusted blindly forever, but this is no longer an unverified
  assumption.
- **Four institutions in the original research (three small closed
  colleges, one for-profit) have no real endowment data to find** — three
  because it was only located after checking earlier filing years than
  initially tried, one (a for-profit) because for-profit institutions do
  not report an endowment field at all, a genuine structural fact rather
  than a gap.
- **The harder half of the coupling machinery (asymmetric, predatory
  resource extraction between containers, as opposed to a clean merger
  or a mutual-benefit arrangement) has one real, worked instance
  (documented in the manuscript) but has not been validated against a
  full retrospective panel fit** — the subordinate institution's own
  chaotic collapse left no single clean container to test against.
- **The `frac_high_entropy` feature cannot currently distinguish a large
  positive shock from a destabilizing one — a real, evidenced limitation,
  found and confirmed 2026-09-15, not yet fixed.** `dynamics.py`'s
  `rolling_causal_variance()` computes plain `.var()` on a channel's
  first differences, and `classify_regime()` compares two of these
  variances to flag "high-entropy" periods; variance is a squared-
  deviation measure, so it is symmetric by construction and cannot
  represent the *direction* of a swing, only its size. Live-scoring
  University of Houston (UNITID 225511) surfaced this directly: its
  `frac_high_entropy` came back 1.0000 (every one of the last 5
  periods flagged high-entropy) against exactly 0.0000 for every
  comparable public flagship already in the validated panel (Michigan,
  UVA, UNC-Chapel Hill, Florida, Wisconsin) — yet Houston's real 2025
  financial condition is the opposite of distressed (S&P upgraded its
  bond rating to AA+, citing a $287M operating surplus and $3.3B in
  reserves), and the university received a real $1.3B infusion from
  the new Texas University Fund in this same period -- a large,
  genuine, *positive* resource shock that this feature has no way to
  tell apart from a debt collapse of the same magnitude. This is not
  a hypothetical: the validated panel itself already contains real
  closures (Green Mountain, Marygrove, MacMurray) sitting at the same
  0.8–1.0 `frac_high_entropy` values Houston now shows, so the fitted
  classifier has no basis in its training data for separating "erratic
  because collapsing" from "erratic because of a sudden windfall."
  This is the same underlying failure mode as the facilities-and-
  athletics-spending and online-class-share proxies that were tested
  and honestly rejected during the original feature-selection work
  (see `reports/RICD Tracker Findings Final.pdf`) — a magnitude-only
  signal that means either thriving or collapse depending on context
  it doesn't have access to — except this instance made it into the
  final 8 validated features rather than being caught beforehand. A
  real fix (giving the regime classifier access to the signed
  direction of a shock, not just its magnitude, likely by cross-
  referencing it against the already-signed debt/reserve features
  rather than replacing it outright) is planned, but has not been
  built or re-validated against the full 54-institution panel yet —
  stated here honestly as open, unfixed work, not quietly patched
  without re-validation.

## Repository structure

````
src/
  model.py               Bayesian state-space model (PyMC)
  jump_diffusion.py       Shock-type latent process for debt
  common_cause.py         Shared-external-shock detector
  real_adapter.py          Real IPEDS/Scorecard data loading (historical, local files)
  fetch_live_data.py       Live data fetching (College Scorecard API + IPEDS bulk files)
  classifier.py             The 8-feature + governance-override classifier
  score_institution.py       End-to-end scoring entry point for one institution (see Known Gaps)
  score_batch.py              Scores a fixed, disclosed list of real institutions in one run (see Known Gaps)
  render_dashboard.py          Bakes docs/data/live_scores.json into docs/index.html as static HTML
  dynamics.py                   Core RICD dynamical-system equations used by the model
  legacy_peer_density_reference.py   Retained reference implementation from an earlier peer-density approach
  diagnose_scorecard_gaps.py    One-off diagnostic: real per-year, per-field College Scorecard completeness for a given institution
  diagnose_feature_values.py    One-off diagnostic: real 8-feature vectors for given institutions vs. real validated-panel reference values (see Known Gaps)
  diagnose_f3_fields.py         One-off diagnostic: confirmed the real F3 (for-profit) finance form's column codes against a live filing
  diagnose_ipeds_access.py      One-off diagnostic: found NCES's real, current bulk-file URL pattern for the newest 1-2 years, after the old one went dead
  diagnose_ipeds_access2.py     One-off diagnostic: found NCES's real, current bulk-file URL pattern for older years (FY2013-FY2021), served from a different location than the newest years
data/
  panel/panel.json           The real, validated 54-institution panel
docs/
  RICD 15.6 master.docx, .tex    The full, domain-independent RICD theory
  index.html                      Public results dashboard (GitHub Pages)
  data/panel.json                  Validated 54-institution panel data
  data/live_scores.json            Real institutions scored live by score_institution.py / score_batch.py
source-documents/
  Quartet of poems.pdf                                                 Original poems
  The Pentagonal Theorem of the Mathematical Nature of Evil.pdf       Became FDFM
  Shaking Bowls Thought Experiment.pdf                                 Became RICS
  Source Translation Ledger source poems explained mathematically.pdf   Poems set line-by-line alongside RICD's math
  README.md                                                               Full lineage
  intermediate-development/
    Feedback Divergence Field Model FDFM U.S. justice system application and research proposal.docx               Early FDFM justice-tracker proposal
    RICS FDFM Multiscale Information Geometric Model.pdf                    Expanded nested RICS-FDFM
    RICD 5.0.pdf, RICD 5.3.pdf, RICD 5.4.pdf,                              Earlier RICD versions
    RICD 5.5.pdf, RICD 5.6.pdf, RICD 1.2 or 1.3 early version.pdf
reports/
  RICD Tracker Findings Final.pdf        Final findings document
  RICD Tracker Narrative Final.pdf       Narrative account of how results were reached
  RICD Tracker Process Log Final.pdf     Consolidated process record
  Higher Ed Sector Findings.pdf           What the results imply about the sector
  Claude's Account of Carmen's Role in Building RICD and the higher-ed tracker.pdf   Claude's own account of the collaboration
  ChatGPT's Account of Its Own Role in the Early Development of FDFM, RICS, and RICD.pdf   ChatGPT's own account of the collaboration
  Actor Tracker Seed Note.pdf             Seed note for a mechanism-layer (actor) tracker, planned for later
  RICD Adapter Instructional Manual.pdf  Adapter-contract implementation guide, for engineers
  RICD Integration Manifest Complete.pdf   Every mechanism confirmed built into RICD, with the manuscript text shown for each
.github/workflows/
  rescore.yml                     Scheduled re-scoring workflow (score_batch.py + score_institution.py + render_dashboard.py)
  diagnose_scorecard_gaps.yml      Manual-only: runs diagnose_scorecard_gaps.py
  diagnose_feature_values.yml      Manual-only: runs diagnose_feature_values.py
  diagnose_f3_fields.yml            Manual-only: runs diagnose_f3_fields.py
  diagnose_ipeds.yml                Manual-only: runs the IPEDS-access diagnostics
````

## Re-scoring cadence

IPEDS is not live data — it releases on a fixed institutional schedule
(provisional data a few times a year, final data annually). The scheduled
workflow in `.github/workflows/rescore.yml` runs periodically and checks
for new data rather than assuming a fixed release date; a run that finds
nothing new is a normal, expected outcome, not a failure.
