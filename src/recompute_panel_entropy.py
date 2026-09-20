"""
Real validation for the directional-entropy fix
(dynamics.rolling_causal_downside_variance, replacing the 2026-09-19
debt_spike-sign gate -- see that function's own docstring and
score_institution.py's SUPERSEDED comment for the full history).

WHY THIS SCRIPT EXISTS: every other fix in this project's history has
been "re-validated against the real 54-institution panel before
shipping, not assumed safe" (see README Known Gaps). The debt_spike
gate could be checked that way with zero network access, because it
only re-derived a value from panel.json's ALREADY-COMPUTED features
(frac_high_entropy, debt_spike). This fix is different: it changes how
sigma_p itself is built from the raw posterior trajectory, which
panel.json never stored (only the final 8-feature vectors are cached
anywhere in this repo -- see recompute_panel_entropy() below and
README Known Gaps on the reset/collapse gap for the same limitation).
So validating this fix for real means re-running the full live
pipeline for every panel institution and refitting the classifier --
which needs real network access to the College Scorecard API and NCES,
which this project's sandboxed development environments do not have
(see fetch_live_data.py's own module docstring). GitHub Actions is the
one environment on this project confirmed able to reach both -- so
that is where this validation has to run, not locally.

WHAT THIS SCRIPT DOES NOT DO: it never writes to the real, trusted
data/panel/panel.json. It writes a separate file,
data/panel/panel_entropy_recomputed.json, and prints a leave-one-out
accuracy comparison (old panel vs. newly-recomputed panel) so a human
can review the real numbers before deciding whether to promote it.
Promoting is a separate, deliberate step (see the end of this
docstring), never automatic.

Sector is not stored in panel.json either (InstitutionFeatures has no
sector field -- only the 8 derived features, name, unitid, outcome,
and governance_flag). This uses fetch_live_data.detect_sector() (the
College Scorecard API's own school.ownership field) to re-derive it
live for each institution, rather than requiring a hand-built 54-row
manifest that could silently go stale.

Real, honest handling of institutions that fail to reproduce: if
sector detection fails, or compute_features_for_institution() returns
None (genuinely insufficient live data -- this pipeline never
fabricates a value to force a result through), the ORIGINAL panel.json
row for that institution is carried over unchanged into the recomputed
file, and it is listed under "COULD NOT RECOMPUTE" in the final
report -- never silently dropped, never guessed.

Uses PRODUCTION sampling settings (n_draws=300, n_tune=300, n_chains=2,
target_accept=0.9 -- score_institution.py's own defaults), not the
higher-precision settings diagnose_window_mismatch.py uses to rule out
MCMC non-convergence. Deliberate: this script validates the fix that
will actually run in production (score_batch.py, rescore.yml), so it
should validate under the same settings production actually uses, not
a different, more expensive configuration nothing else in this
pipeline runs with.

Only START_YEAR_OVERRIDES below is a known, confirmed exception
(University of Phoenix-Arizona needs 2014 -- see score_batch.py's own
docstring for why: 2013 returned a genuinely empty College Scorecard
record for this UNITID). Every other panel institution defaults to
2013; if any of the other 53 come back insufficient_data at that
default, that is real information this script's report surfaces
(under COULD NOT RECOMPUTE) rather than a reason to silently guess a
different start_year for it.

Usage:
    cd src
    python recompute_panel_entropy.py
Requires COLLEGE_SCORECARD_API_KEY in the environment (same as
score_institution.py / score_batch.py / diagnose_window_mismatch.py).
Expect a long run: up to 54 live data fetches plus Bayesian fits, each
comparable in time to one score_batch.py institution.

NEXT STEP AFTER A CLEAN RUN -- NOT AUTOMATIC: review the leave-one-out
accuracy comparison this script prints. If it holds (ideally at
100.00%, matching the current panel), promote the recomputed panel
yourself:
    mv data/panel/panel_entropy_recomputed.json data/panel/panel.json
Only after that promotion does it make sense to re-run score_batch.py
(or wait for the next scheduled rescore.yml) so Houston, UCF, FSU,
Buffalo, Clemson, and Cal State Long Beach get scored against the
classifier refit on the corrected panel, rather than just against the
corrected live-scoring code alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from classifier import RICDClassifier, load_panel
from score_institution import compute_features_for_institution
import fetch_live_data as fld

PANEL_PATH = "../data/panel/panel.json"
OUTPUT_PATH = "../data/panel/panel_entropy_recomputed.json"

# Confirmed, known exception -- see score_batch.py's own docstring: 2013
# returned a genuinely empty College Scorecard record for this UNITID.
# Not a guess; every other panel institution defaults to 2013 below.
START_YEAR_OVERRIDES = {
    "484613": 2014,  # University of Phoenix-Arizona
}


def main():
    original_panel = load_panel(PANEL_PATH)
    original_by_id = {inst.unitid: inst for inst in original_panel}

    new_rows = []
    could_not_recompute = []

    for inst in original_panel:
        unitid, name = inst.unitid, inst.name
        print(f"\n{'=' * 70}\nRecomputing {name} ({unitid})\n{'=' * 70}")

        sector = fld.detect_sector(unitid)
        if sector is None:
            print(f"COULD NOT DETECT SECTOR for {name} ({unitid}) -- "
                  f"carrying over its original panel row unchanged.")
            could_not_recompute.append((unitid, name, "sector detection failed"))
            new_rows.append(inst)
            continue

        start_year = START_YEAR_OVERRIDES.get(unitid, 2013)
        try:
            features = compute_features_for_institution(unitid, name, sector=sector, start_year=start_year)
        except Exception as e:
            print(f"REAL ERROR recomputing {name} ({unitid}): {type(e).__name__}: {e} -- "
                  f"carrying over its original panel row unchanged.")
            could_not_recompute.append((unitid, name, f"{type(e).__name__}: {e}"))
            new_rows.append(inst)
            continue

        if features is None:
            print(f"INSUFFICIENT LIVE DATA for {name} ({unitid}) at start_year={start_year} -- "
                  f"carrying over its original panel row unchanged.")
            could_not_recompute.append((unitid, name, f"insufficient_data at start_year={start_year}"))
            new_rows.append(inst)
            continue

        # Ground truth (real confirmed outcome, governance flag) is never
        # re-derived -- only the 8 model-derived features are replaced.
        features.outcome = inst.outcome
        features.governance_flag = inst.governance_flag
        new_rows.append(features)

        old_feh = original_by_id[unitid].frac_high_entropy
        print(f"frac_high_entropy: {old_feh:.4f} (old, symmetric-variance panel value) -> "
              f"{features.frac_high_entropy:.4f} (new, downside-variance recompute)")

    Path(OUTPUT_PATH).write_text(json.dumps([row.__dict__ for row in new_rows], indent=2))
    print(f"\nWrote {len(new_rows)} rows to {OUTPUT_PATH}")

    print(f"\n{'=' * 70}\nVALIDATION: leave-one-out accuracy, old panel vs. recomputed panel\n{'=' * 70}")
    clf_old = RICDClassifier()
    acc_old, misclassified_old = clf_old.leave_one_out_accuracy(original_panel)
    clf_new = RICDClassifier()
    acc_new, misclassified_new = clf_new.leave_one_out_accuracy(new_rows)
    print(f"OLD data/panel/panel.json          : {acc_old:.2%} accuracy, misclassified: {misclassified_old}")
    print(f"NEW panel_entropy_recomputed.json  : {acc_new:.2%} accuracy, misclassified: {misclassified_new}")

    if could_not_recompute:
        print(f"\n{'=' * 70}\nCOULD NOT RECOMPUTE ({len(could_not_recompute)} institutions -- "
              f"original row carried over unchanged for each)\n{'=' * 70}")
        for unitid, name, reason in could_not_recompute:
            print(f"  {name} ({unitid}): {reason}")

    print(f"\n{'=' * 70}\nNEXT STEP -- NOT AUTOMATIC: review the accuracy comparison above. "
          f"If it holds, promote the recomputed panel yourself:\n"
          f"  mv data/panel/panel_entropy_recomputed.json data/panel/panel.json\n"
          f"Then re-run score_batch.py (or wait for the next scheduled rescore.yml) so "
          f"Houston, UCF, FSU, Buffalo, Clemson, and Cal State Long Beach get scored "
          f"against a classifier refit on the corrected panel.\n{'=' * 70}")


if __name__ == "__main__":
    main()
