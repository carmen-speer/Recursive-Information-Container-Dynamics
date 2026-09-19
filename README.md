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
- **The `frac_high_entropy` feature could not distinguish a large positive
  shock from a destabilizing one — found 2026-09-15, gated (not fully
  resolved) 2026-09-19.** `dynamics.py`'s `rolling_causal_variance()`
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
  **Fix applied 2026-09-19** in `score_institution.py`: `frac_high_entropy`
  is now gated on `debt_spike`, the one already-signed feature already in
  the vector — zeroed whenever `debt_spike <= 0`, i.e. whenever the
  volatility isn't coming from rising liabilities. Re-validated against
  the real panel, not assumed safe: applying the gate to the 54
  institutions' already-computed features changes exactly 2 (Spelman,
  Clark Atlanta — both real confirmed-stable, both previously flagged
  from debt *declining*, not spiking), and leave-one-out accuracy on the
  corrected panel remains 100.00% (54/54) — every real closure with high
  `frac_high_entropy` in the panel also has `debt_spike > 0`, so none
  lose their signal. **What this fix has not yet been shown to do:**
  produce corrected live scores for Houston, UCF, FSU, Buffalo, or
  Clemson — that requires re-running the live pipeline against real,
  freshly-fetched data, which needs network access and a live
  `COLLEGE_SCORECARD_API_KEY` neither available in this development
  environment (see the NCES access note above) nor exercised since this
  change. The dashboard and its interpretation note should not be
  described as "fixed" for those institutions until a real re-score run
  confirms it. West Virginia University and Sweet Briar's underlying
  cause has not been checked against this specific mechanism at all
  (Sweet Briar's 52.4% is believed to be a *different* gap — see the
  reset/recovery item below, not this one). This gate is also a coarse
  first cut, not a final design: it only checks the sign of one already-
  signed feature rather than giving the entropy measure its own genuine
  directional construction, and it does nothing for the second, separate
  gap below.
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

## Repository structure
