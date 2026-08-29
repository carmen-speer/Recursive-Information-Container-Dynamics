# RICD Higher-Education Collapse Tracker

A Bayesian state-space model and validated classifier for institutional
financial collapse risk in U.S. higher education, built on the Recursive
Information-Container Dynamics (RICD) framework.

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

- **The live-scoring pipeline (`src/score_institution.py`) is not yet
  fully wired end to end.** Live data fetching from the College Scorecard
  API and IPEDS bulk finance files works; the step that runs that fresh
  data through the Bayesian model to produce real feature values for a
  *new* institution (as opposed to the already-fitted historical
  trajectories in the validation panel) is the next concrete piece of
  work, not a stub with fabricated numbers standing in for it.
- **IPEDS finance data has no stable API.** `src/fetch_live_data.py`
  downloads bulk files using a URL and field-code pattern confirmed
  during this project's original build, but NCES has changed this
  pattern before (there was a real transition in reporting standards
  around 2018) and may again. Automated re-scoring should be monitored,
  not trusted blindly, on this specific point.
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
