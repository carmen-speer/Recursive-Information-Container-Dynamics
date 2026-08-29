# RICD Higher-Education Collapse Tracker

A Bayesian state-space model and validated classifier for institutional
financial collapse risk in U.S. higher education, built on the Recursive
Information-Container Dynamics (RICD) framework.

**The full RICD manuscript (the complete, domain-independent theory) lives at
[`manuscript/RICD_10_7_manuscript.pdf`](manuscript/RICD_10_7_manuscript.pdf).**
Everything in this repository's code implements a real subset of that
framework (Parts 6, 7, 8, and 10.5b specifically) against U.S. higher-education
data; the manuscript itself is domain-independent and covers considerably more
than the tracker uses.

**The original source documents RICD was built from live in
[`source-documents/`](source-documents/):** the quartet of poems, the
Pentagonal Theorem of the Mathematical Nature of Evil, and Shaking Bowls —
Carmen Speer's own original creative and theoretical writing, which became
FDFM and RICS respectively, then nested into RICS-FDFM, then RICD, before
107 further revisions arrived at the manuscript above. The folder's
`intermediate-development/` subfolder holds real, surviving milestones from
that path: an early FDFM application proposing a justice-system tracker
(directly cited by RICD 5.0's own editorial notes, not a lost document), the
expanded RICS-FDFM formalization, and five earlier RICD versions (5.0
through 5.6). See that folder's own README for the full lineage.

**The narrative, findings, and process record of how this tracker was built
live in [`reports/`](reports/):** the findings document, the narrative account
of how each result was actually reached, the consolidated process log, a
separate analysis of what the results imply about the U.S. higher-education
sector as a whole, a record of Carmen Speer's role in building the framework
and the tracker, a roadmap for planned future trackers in other domains, an
instructional manual for engineers working with the RICD adapter contract
directly, and a seed note for a genuinely different kind of tracker planned
for later: an
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

```bash
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
```

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

- **The live-scoring pipeline (`src/score_institution.py`) is now fully
  wired end to end, but genuinely untested against live data.** It fetches
  real enrollment/finance data, runs it through the same Bayesian model
  and feature computation the validated 54-institution panel uses, and
  produces a real classification. It could not be exercised end-to-end
  from the sandboxed environment this was built in, since `nces.ed.gov`
  is not in that sandbox's own network allowlist — confirmed directly (the
  same block was returned for old and new URL patterns alike, dressed up
  as an HTTP 403 by both `curl` and Python's `requests`, which is itself
  a real lesson: don't conclude a remote service changed behavior until
  you've ruled out your own environment first). This should work in a
  normal environment (GitHub Actions, a local machine) without that
  specific restriction, but has not been confirmed working there yet —
  treat it as wired, not as validated.
- **IPEDS finance data has no stable API, and its distribution mechanism
  has already changed once during this project.** The original bulk-file
  URL pattern (`nces.ed.gov/ipeds/datacenter/data/F....zip`) is now
  defunct; `src/fetch_live_data.py` has been updated to the current,
  real endpoint (`nces.ed.gov/ipeds/data-generator?...`, confirmed
  directly against NCES's own live Complete Data Files page), which
  also returns a raw CSV rather than a zip archive. NCES has changed
  this once already and may again — automated re-scoring should be
  monitored, not trusted blindly, on this specific point.
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

## Repository structure

```
src/
  model.py               Bayesian state-space model (PyMC)
  jump_diffusion.py       Shock-type latent process for debt
  common_cause.py         Shared-external-shock detector
  real_adapter.py          Real IPEDS/Scorecard data loading (historical, local files)
  fetch_live_data.py       Live data fetching (College Scorecard API + IPEDS bulk files)
  classifier.py             The 8-feature + governance-override classifier
  score_institution.py       End-to-end scoring entry point (see Known Gaps)
data/
  panel/panel.json           The real, validated 54-institution panel
manuscript/
  RICD_10_7_manuscript.pdf    The full, domain-independent RICD theory
source-documents/
  Quartet_of_poems.pdf                                            Original poems
  The_Pentagonal_Theorem_of_the_Mathematical_Nature_of_Evil_-2.pdf  Became FDFM
  Shaking_Bowls_Thought_Experiment-1.pdf                            Became RICS
  README.md                                                          Full lineage
  intermediate-development/
    Feedback_Divergence_Field_Model...justice_system....docx          Early FDFM justice-tracker proposal
    RICS_FDFM_Multiscale_Information_Geometric_Model.pdf               Expanded nested RICS-FDFM
    RICD_5_0.docx, RICD_5_3.pdf, RICD_5_4.pdf,                         Six earlier RICD versions
    RICD_5_5.docx, RICD_5_6.pdf, RICD_1_2_or_1_3_early_version.pdf
reports/
  RICD_Tracker_Findings.docx        Final findings document
  RICD_Tracker_Narrative.docx       Narrative account of how results were reached
  RICD_Tracker_Process_Log.docx     Consolidated process record
  Higher_Ed_Sector_Findings.docx    What the results imply about the sector
  Carmens_Role.docx                  Carmen Speer's role in building RICD and the tracker
  Future_Trackers_Roadmap.docx       Planned future trackers in other domains
  Actor_Tracker_Seed_Note.docx       Seed note for a mechanism-layer (actor) tracker, planned for later
  RICD_Adapter_Instructional_Manual.docx   Adapter-contract implementation guide, for engineers
results/
  latest_scores.json         Most recent scoring run (populated by the scheduled workflow)
docs/
  index.html                 Simple public results dashboard (GitHub Pages)
.github/workflows/
  rescore.yml                 Scheduled re-scoring workflow
```

## Re-scoring cadence

IPEDS is not live data — it releases on a fixed institutional schedule
(provisional data a few times a year, final data annually). The scheduled
workflow in `.github/workflows/rescore.yml` runs periodically and checks
for new data rather than assuming a fixed release date; a run that finds
nothing new is a normal, expected outcome, not a failure.
