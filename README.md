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
engineers working with the RICD adapter contract directly), and the
complete [`RICD Integration Manifest Complete.pdf`](<reports/RICD Integration Manifest Complete.pdf>) described above.

**Not-yet-built or not-yet-funded work lives in [`future-projects/`](future-projects/):**
[`Next Project for RICD - Four Adapter Build Roadmap.pdf`](<future-projects/Next Project for RICD - Four Adapter Build Roadmap.pdf>)
(a proposed one-year build plan for four new RICD tracker adapters --
banking, hospitals, public schools, and nursing homes/long-term care --
chosen for having both large public datasets and realistic commercial
application, prepared for grant application documentation),
[`RICD Ongoing Projects Roadmap.pdf`](<future-projects/RICD Ongoing Projects Roadmap.pdf>)
(the broader roadmap of further RICD-based trackers planned beyond the four
adapters and the higher-ed tracker in this repository),
[`Collapse-Causal Agent Tracker (C-CAT) Seed Note.pdf`](<future-projects/Collapse-Causal Agent Tracker (C-CAT) Seed Note.pdf>)
(a seed note for a genuinely different kind of tracker: an agent tracker,
investigating the specific real decisions and agents behind a collapse --
board minutes, depositions, investigative findings -- rather than the
aggregate financial and enrollment effects the higher-ed tracker measures),
[`Plurimo Vision Document.pdf`](<future-projects/Plurimo Vision Document.pdf>)
(a teaching-optimization software platform for private tutors and
independent educators that uses RICD directly in its own algorithm -- for
site-health monitoring and its anti-capture/anti-trolling detection system
-- which is exactly why RICD needs to be pressure-tested and refined
against real, adversarial data across new domains, via the four adapters
above, before it's trusted to run Plurimo's systems; Plurimo is planned to
be funded by revenue from those same four adapters rather than sought
directly), and
[`Web–Field Model of Cognitive Trait Distributions and Social Evolution.pdf`](<future-projects/Web–Field Model of Cognitive Trait Distributions and Social Evolution.pdf>)
(RICD's mathematics applied to a second, unrelated domain: modeling human
cognitive traits as regions on a continuous manifold rather than discrete
types, and deriving from it a proposed learning-style metric, the Cognitive
Trait Manifold Ratio, or CMTR -- a working theory, at an earlier stage than
anything else in this repository, included so its actual state is checkable
rather than asserted after the fact once, or if, it's validated).

The planned future tracker mentioned above is an
**agent tracker**, which would investigate the specific real decisions and
agents behind a collapse (board minutes, depositions, investigative
findings), rather than the aggregate financial and enrollment effects this
higher-ed tracker measures. The two are complementary, not competing —
see the seed note itself for why an agent tracker needs a different kind of
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
- **The `frac_high_entropy` feature could not distinguish a large positive
  shock from a destabilizing one — found 2026-09-15, fixed at the source
  and validated 2026-09-20, with new open questions from that fix's own
  live re-score (see the end of this item).**
  `dynamics.py`'s `rolling_causal_variance()`
  computes plain `.var()` on a channel's first differences, and
  `classify_regime()` compares two of these variances to flag
  "high-entropy" periods; variance is a squared-deviation measure, so
  it is symmetric by construction and cannot represent the *direction*
  of a swing, only its size. Live-scoring University of Houston (UNITID
  225511) surfaced this directly: its `frac_high_entropy` came back
  1.0000 (every one of the last 5 periods flagged high-entropy) against
  exactly 0.0000 for every comparable public flagship already in the
  validated panel (Michigan, UVA, UNC-Chapel Hill, Florida, Wisconsin)
  — yet Houston's real 2025 financial condition is the opposite of
  distressed (S&P upgraded its bond rating to AA+, citing a $287M
  operating surplus and $3.3B in reserves), and the university received
  a real $1.3B infusion from the new Texas University Fund in this same
  period. Live-scoring a further batch (UCF, FSU, University at Buffalo,
  Clemson, Cal State Long Beach) surfaced the same shape of problem: UCF
  and FSU both carry real, confirmed stable bond ratings (Florida Board
  of Governors filing, 2/27/26: UCF Moody's Aa2/Fitch AA stable, FSU
  Moody's Aa1/Fitch AA+ stable, neither on negative outlook) directly
  contradicting high-risk scores driven substantially by this same
  feature. This is not a hypothetical: the validated panel itself
  already contains real closures (Green Mountain, Marygrove, MacMurray)
  sitting at the same 0.8–1.0 `frac_high_entropy` values Houston shows,
  so the fitted classifier had no basis in its training data for
  separating "erratic because collapsing" from "erratic because of a
  sudden windfall." Same underlying failure mode as the facilities-and-
  athletics-spending and online-class-share proxies that were tested
  and honestly rejected during the original feature-selection work
  (see `reports/RICD Tracker Findings Final.pdf`) — a magnitude-only
  signal that means either thriving or collapse depending on context it
  doesn't have access to — except this instance made it into the final
  8 validated features rather than being caught beforehand.
  **First fix (2026-09-19, since superseded): gated `frac_high_entropy`
  on `debt_spike`'s sign** — zeroed whenever `debt_spike <= 0`. Checked
  against the panel's already-computed features (no live re-fit needed
  for that check) and looked safe: changed exactly 2 of 54 institutions
  (Spelman, Clark Atlanta — both real confirmed-stable, previously
  flagged from debt *declining*, not spiking), leave-one-out accuracy
  held at 100.00%. **Proven insufficient by a real, live re-score run**
  (2026-09-19/20, after the gate was already deployed): Houston, UCF,
  FSU, Buffalo, Clemson, and Cal State Long Beach all still scored
  `high_risk` at essentially the same probabilities as before the gate
  existed — the gate never fired for them, most likely because their own
  `debt_spike` came back *positive* (ordinary capital-project borrowing,
  not distress), which this coarse sign-check cannot distinguish from
  genuine debt-driven stress.
  **Second fix, applied 2026-09-20** in `dynamics.py` /
  `score_institution.py`: the debt_spike gate is removed, and
  `frac_high_entropy` is now built from a genuinely directional
  entropy measure at its source — `dynamics.rolling_causal_downside_variance()`
  replaces plain symmetric variance for the operational channel, so only
  *declines* (not any large swing) count toward high-entropy
  classification, independent of `debt_spike` entirely (see that
  function's own docstring for the full reasoning, including a synthetic
  sanity check confirming a positive shock now scores zero while an
  equivalent negative shock does not). `debt_spike` itself is untouched
  and still stored as its own independent feature — it no longer gates
  anything. **FIXED and validated, 2026-09-20:** unlike the first fix,
  this one changes how `frac_high_entropy` is computed from raw
  posterior trajectories, which the panel never cached (only the final
  8-feature vectors are stored anywhere in this repo), so it needed a
  live re-fit of all 54 institutions to check — `src/recompute_panel_entropy.py`
  and its matching manual-only workflow did exactly that on GitHub
  Actions (the one environment on this project confirmed able to reach
  NCES/College Scorecard). Result: leave-one-out accuracy holds at
  100.00% on the refit panel, zero misclassifications — the recomputed
  panel was promoted to `data/panel/panel.json` the same night. A fresh
  live re-score of Houston/UCF/FSU/Clemson/Long Beach was also run
  against the fix. Real result, not fully clean either way: Houston —
  the case that motivated this whole fix — is resolved, now scoring
  `stable` (40.6%), consistent with its real AA+ bond rating and $3.3B
  reserves. But two new, unexplained results surfaced in the same run,
  not smoothed over: Clemson (39.0%) and West Virginia University
  (44.2%) both flipped to `stable` under this fix despite each carrying
  real, independently documented financial distress (Clemson's $2.65B
  in long-term liabilities, up $231.9M year-over-year; West Virginia's
  real 2023 financial crisis and program/faculty cuts) — whether the
  new downside-only entropy measure has swung too far the other
  direction for these two, suppressing a real signal it used to
  (over)detect, is now the open question, not yet resolved either way.
  UCF (83.4%, down from 95.4%) and FSU (57.1%, down from 87.1%) are
  still `high_risk` but meaningfully lower than before the fix — partial
  movement that hasn't been explained yet. Cal State Long Beach (95.8%)
  and Sweet Briar (80.1%) remain `high_risk`; Sweet Briar's case is
  still believed to be the separate reset/recovery gap below, not this
  one. University of Phoenix-Arizona remains `high_risk` at effectively
  100% (99.997%). University at Buffalo was removed from the live batch
  entirely after this fix — see `score_batch.py`'s own docstring: its
  score sat at a near-50/50 that didn't map to any mechanism this
  model's features are built to detect (its real strain is a $47M
  federal research-funding cut).
- **The model has no separate state for "collapse, but already reset" —
  found 2026-09-19, not yet built.** Every feature currently in the
  vector is a function of an institution's *most recent* observed
  window (`frac_high_entropy` looks at the last 5 periods; `d_A_trend`
  and `delta_R_trend` compare a mid-period average to a late-period
  one). None of them ask whether a large disruption is still in
  progress or has already been absorbed. University of Phoenix-Arizona
  scoring 100.0% is the live case: it underwent a real, large
  enrollment contraction and restructuring, and the working hypothesis
  (not yet confirmed against its real data, which this environment
  can't fetch) is that the model is reading the *aftermath* of a
  completed reset as ongoing collapse, because both look identical to
  every feature currently computed only over the tail window. Sweet
  Briar College (52.4%) is the panel's own version of this same
  question, and the reason it's the more useful test case: it had a
  real, documented near-closure and recovery in 2015, so a real
  post-recovery time series already exists for it, unlike Phoenix.
  A scoped fix, not yet built or validated: add a *within-window*
  trajectory feature that compares entropy/divergence in the earlier
  part of the current lookback window against the most recent 1-2
  periods specifically — high-then-declining reads as stabilizing
  after a shock, high-and-still-rising reads as ongoing collapse. This
  cannot be built or checked from cached data: it needs the raw,
  per-period `O_o`/`O_p` posterior trajectories for Phoenix and for
  Sweet Briar's full historical series (2010-present, spanning its
  crisis and recovery), neither of which is stored anywhere in this
  repository — only the final 8-feature vectors are (`data/panel/panel.json`,
  `docs/data/live_scores.json`). Getting Sweet Briar's real trajectory
  through its 2015 crisis and recovery, and checking whether a
  within-window trend feature would have called that recovery
  correctly, is the concrete next experiment, not a hypothetical one.
- **`reserve_adequacy` — confirmed the classifier's single strongest
  feature (largest-magnitude fitted coefficient, -1.88 on the
  standardized panel, next closest -1.38 for `d_A_trend`) — has no
  real distress example to calibrate against anywhere in the public
  sector, and this is a structural gap in the data that exists, not a
  data-collection failure in this project's pipeline.** The validated
  panel's five public institutions (Michigan, UVA, UNC-Chapel Hill,
  Florida, Wisconsin) are all real, confirmed-stable outcomes, sitting
  at `reserve_adequacy` 10.89–12.93 against a panel-wide median of
  11.62 — not extreme outliers within this panel, but all five were
  selected into the original research specifically for being large,
  wealthy, well-documented flagships, not as a representative sample
  of the roughly 600 public four-year institutions nationally, most of
  which carry far less endowment per student than any of the five.
  There is no confirmed-closure public institution in the panel to set
  against them, and this is not something this project failed to find:
  Kelchen, Ritter & Webber, "Predicting College Closures and Financial
  Distress" (Federal Reserve Bank of Philadelphia, WP 24-20, Dec.
  2024) — a national study covering 2002-2023 — found only two
  four-year public closures in that entire 21-year period nationally
  (one tribal college, one graduate health-sciences-focused
  institution, neither a broad-based comprehensive or flagship public
  university), and excluded public institutions from its own
  closure-prediction model for exactly this reason: too rare a base
  rate to fit against. The paper's own explanation is direct: "Closing
  a public college is a deeply political decision, similar to closing
  a military base," so states use mergers and consolidations instead
  of closure when a public institution is in real distress (a live,
  current example: East Georgia State College's 2026 merger into
  Georgia Southern University, not a closure by this project's outcome
  definition). **Practical consequence: `reserve_adequacy`-driven
  scores for any public institution that is not itself an unusually
  well-endowed flagship — Houston, UCF, FSU, Buffalo, Clemson, Cal
  State Long Beach, and any future one like them — are extrapolating
  outside the range of outcomes this feature was ever validated
  against for that sector, on both sides: no distress example, and no
  ordinary-wealth stable example either.** This is not expected to
  resolve on its own. Fixing it for real needs either a genuine new
  confirmed public four-year closure to add to the panel (rare by the
  numbers above, not something to wait on) or a deliberate, disclosed
  decision to widen this project's own outcome definition for the
  public sector specifically — e.g., treating a distress-driven merger
  or consolidation as the public-sector analogue of a private closure
  — and that would be a real change to what "confirmed outcome" means
  in this project, not a quiet patch, so it should be made openly and
  argued for on its own, not slipped in to make an inconvenient gap
  disappear.
  **One real future candidate, not yet usable: Penn State York.**
  Checked against three other candidates an AI search surfaced
  (2026-09-20) — IUPUI's 2024 split into IU Indianapolis and Purdue
  Indianapolis, and UT Brownsville's 2015 dissolution into UT Rio
  Grande Valley, were both ruled out on the merits: IUPUI's own
  announcement frames the split as strategic expansion, not distress
  ("dramatically growing needs of our state," new investment
  pledged by both universities, no financial or enrollment rationale
  at all); UT Brownsville was a real financial-distress case but
  resolved the same way East Georgia State College did above — a
  rescue-by-merger with the campus, students, and faculty carried
  into the new institution, plus an affirmative strategic upside
  (Permanent University Fund access, a new medical school) — not a
  termination. Penn State York is different in kind: real, cited
  financial losses, 61% enrollment decline from its peak (703
  students, Fall 2024), $29.9M in deferred maintenance, explicitly
  announced (May 2025) as closing for exactly those reasons, not a
  merger. It is not yet usable for two independent reasons, not one:
  first, it hasn't happened yet — Penn State York is set to close
  after the Spring 2027 semester, and this project's outcome
  standard is a completed, confirmed event, not a scheduled one;
  second, and unresolved as of this writing, Penn State York is a
  commonwealth campus operating under Pennsylvania State University's
  single overall accreditation, not a separately accredited
  institution, and it is not yet confirmed whether IPEDS carries
  separate institution-level Finance data for it or whether its
  finances are consolidated into Penn State's university-wide filing
  — the latter would leave nothing for this project's per-institution
  pipeline to extract, since Penn State as a whole is a stable,
  thriving R1 university, not a closing one. Confirming the IPEDS
  finance-reporting question needs live network access this
  development environment doesn't have (same limitation noted
  throughout this section); revisiting Penn State York after Spring
  2027, once its closure is a completed fact rather than an
  announced plan, is the concrete next check, not something to
  chase down early.

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
  diagnose_panel_pipeline_consistency.py   One-off diagnostic: checks the panel-loading pipeline's path resolution and data consistency
  diagnose_window_mismatch.py    One-off diagnostic: separates a real live-vs-panel data-window mismatch from ordinary MCMC non-convergence
  recompute_panel_entropy.py     Real validation for the directional-entropy fix: re-fits all 54 panel institutions live and compares leave-one-out accuracy (see Known Gaps)
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
  Collapse-Causal Tracker (C-CT) Description Document.pdf   What a collapse-causal tracker is as a category, and how it differs from C-CAT
  RICD Adapter Instructional Manual.pdf  Adapter-contract implementation guide, for engineers
  RICD Integration Manifest Complete.pdf   Every mechanism confirmed built into RICD, with the manuscript text shown for each
  readme.md                                Guide to this folder's contents
future-projects/
  Collapse-Causal Agent Tracker (C-CAT) Seed Note.pdf         Seed note for a mechanism-layer (agent) tracker, planned for later
  Next Project for RICD - Four Adapter Build Roadmap.pdf      One-year build plan for four new tracker adapters (banking, hospitals, public schools, nursing homes/long-term care)
  Plurimo Vision Document.pdf                                  Vision document for Plurimo, a teaching-optimization platform that uses RICD in its own algorithm
  RICD Ongoing Projects Roadmap.pdf                            Broader roadmap of further planned RICD-based trackers
  Web–Field Model of Cognitive Trait Distributions and Social Evolution.pdf   CMTR working theory (see above)
.github/workflows/
  rescore.yml                     Scheduled re-scoring workflow (score_batch.py + score_institution.py + render_dashboard.py)
  claude.yml                       Claude Code GitHub Action (handles @claude-triggered edits made directly on GitHub)
  diagnose_scorecard_gaps.yml      Manual-only: runs diagnose_scorecard_gaps.py
  diagnose_feature_values.yml      Manual-only: runs diagnose_feature_values.py
  diagnose_f3_fields.yml            Manual-only: runs diagnose_f3_fields.py
  diagnose_ipeds.yml                Manual-only: runs the IPEDS-access diagnostics
  diagnose_panel_pipeline_consistency.yml   Manual-only: runs diagnose_panel_pipeline_consistency.py
  diagnose_window_mismatch.yml      Manual-only: runs diagnose_window_mismatch.py
  recompute_panel_entropy.yml        Manual-only: runs recompute_panel_entropy.py (see Known Gaps)
````

## Re-scoring cadence

IPEDS is not live data — it releases on a fixed institutional schedule
(provisional data a few times a year, final data annually). The scheduled
workflow in `.github/workflows/rescore.yml` runs periodically and checks
for new data rather than assuming a fixed release date; a run that finds
nothing new is a normal, expected outcome, not a failure.
