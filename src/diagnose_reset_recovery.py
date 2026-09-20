"""
Diagnostic for the "collapse, but already reset" gap documented in
README Known Gaps (found 2026-09-19, not yet built). Every one of the
eight validated features looks only at an institution's most recent
observed window (frac_high_entropy specifically at its last 5
periods) -- none of them ask whether a disruption visible earlier in
that same window is still worsening or has since stabilized.

University of Phoenix-Arizona (100.0% live) and Sweet Briar College
(80.1% live, 52.4% in the frozen panel) are this project's two live
test cases for that gap. Sweet Briar is the more useful one because it
has a REAL, CONFIRMED post-crisis recovery (announced closure 2015,
reopened after an alumnae-led rescue) to check a candidate fix
against, where Phoenix's restructuring outcome is not itself
independently confirmed one way or the other.

WHAT THIS SCRIPT DOES: computes, for both institutions, the concrete
within-window trajectory feature the README already scopes as the
next experiment (not a new idea invented here) -- splits each
institution's live lookback window into an EARLY segment (the
mid-window slice d_A_trend already uses elsewhere in this pipeline)
and a LATE segment (the most recent 5 periods -- the same window
frac_high_entropy already uses), and computes the downside-entropy-
based high-entropy fraction separately for each segment.

  within_window_trend = frac_high_entropy(LATE) - frac_high_entropy(EARLY)

A meaningfully POSITIVE value (late higher than early) reads as
entropy that is still rising within the window -- ongoing collapse. A
meaningfully NEGATIVE value (late lower than early) reads as entropy
that peaked earlier in the window and has since fallen -- stabilizing
after a shock, exactly what a real post-crisis recovery should look
like.

WHY SWEET BRIAR IS THE REAL TEST, NOT JUST A PLAUSIBLE STORY: Sweet
Briar's window is deliberately set to start at 2010 (not the panel/
live-batch default of 2013) specifically so it spans its real 2015
near-closure and its real post-2016 recovery. If this candidate
feature is doing what it's meant to, Sweet Briar's within_window_trend
should come back meaningfully NEGATIVE -- a real, falsifiable
prediction. If it instead comes back flat or positive, that is real
evidence against this specific construction, not a reason to force it
into the classifier anyway.

WHY PHOENIX IS NOT A CONFIRMATORY TEST THE SAME WAY: Phoenix's real,
large enrollment contraction and restructuring is well documented, but
whether that restructuring reads as a completed, stabilizing reset as
of the live pipeline's current window, or an ongoing decline, is not
itself an independently confirmed fact this project has. Phoenix's
result here is exploratory, read alongside Sweet Briar's, not a second
independent confirmation of the same kind.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO: it does not add
within_window_trend to classifier.FEATURE_NAMES, does not touch
dynamics.py, classifier.py, or score_institution.py, and does not
refit or re-validate the classifier against the panel. Promoting an
unvalidated 9th feature into the live classifier without a real
leave-one-out re-check against the full panel is exactly the kind of
unverified change this project does not make (see README Known Gaps
throughout, and recompute_panel_entropy.py's own docstring for the
precedent this would have to follow if this candidate holds up). This
script's only job is to produce the real numbers needed to decide
whether that further step is worth taking at all.

Requires COLLEGE_SCORECARD_API_KEY in the environment (same as every
other live-data script in this project) -- run via GitHub Actions, not
locally: this project's sandboxed development environments cannot
reach NCES/College Scorecard (see fetch_live_data.py's own module
docstring).

Does NOT call save_live_score() -- never touches
docs/data/live_scores.json or the public dashboard. Diagnostic only.

Usage:
    python diagnose_reset_recovery.py
"""

from __future__ import annotations

import datetime
import traceback

import numpy as np
import pymc as pm

import model as mdl
import dynamics as dyn
import real_adapter as ra
import fetch_live_data as fld

# Sweet Briar: 2010, not the panel/live-batch default of 2013, so the
# window actually includes its real 2015 crisis rather than starting
# two years into it. University of Phoenix-Arizona keeps its known
# start_year=2014 override -- see score_batch.py's own docstring: 2013
# returned a genuinely empty College Scorecard record for this UNITID.
TARGETS = [
    ("233718", "Sweet Briar College", "private", 2010),
    ("484613", "University of Phoenix-Arizona", "forprofit", 2014),
]

# Production sampling settings (score_institution.py's own defaults) --
# this validates what would actually run in production, not a
# different, more expensive configuration nothing else in this
# pipeline runs with.
N_DRAWS, N_TUNE, N_CHAINS, TARGET_ACCEPT = 300, 300, 2, 0.9

# Within-window_trend magnitude below this (in absolute value) is read
# as "no real within-window movement either way", not as a false
# negative -- avoids over-reading small-sample noise as a clean result.
FLAT_THRESHOLD = 0.05


def compute_within_window_trend(unitid: str, name: str, sector: str, start_year: int,
                                 end_year: int | None = None) -> dict | None:
    end_year = end_year or (datetime.date.today().year - 2)

    series = fld.build_live_series(unitid, start_year, end_year, sector=sector)
    if series is None:
        print(f"INSUFFICIENT ENROLLMENT DATA for {name} ({unitid}) across {start_year}-{end_year}. "
              f"Not scoring rather than guessing.")
        return None

    O_o_real, O_p_real = ra.compute_O_o_O_p(series)
    types = ["observed"] * len(O_o_real)

    dest_base = f"/tmp/ipeds_live/{unitid}"
    finance_years_available = []
    for year in range(start_year, end_year):
        result = fld.download_ipeds_finance_bulk(year, dest_dir=dest_base, sector=sector)
        if result:
            finance_years_available.append(year)
    if len(finance_years_available) < 3:
        print(f"INSUFFICIENT FINANCE DATA for {name} ({unitid}): only "
              f"{len(finance_years_available)} real years downloaded. Not scoring rather than guessing.")
        return None

    window_years = [f"{y}-{str(y + 1)[2:]}" for y in range(start_year, end_year)]
    E_exch, M_maint, W_instr, W_total, mask, scale = fld.parse_live_finance(
        unitid, window_years, dest_base, sector=sector)
    if mask.sum() < 3:
        print(f"INSUFFICIENT PARSED FINANCE DATA for {name} ({unitid}): only "
              f"{int(mask.sum())} real years parsed. Not scoring rather than guessing.")
        return None

    pymc_model = mdl.build_model_stage2(O_o_real, O_p_real, types, types, E_exch, M_maint, W_instr, W_total, mask)
    with pymc_model:
        idata = pm.sample(N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=N_CHAINS,
                           target_accept=TARGET_ACCEPT, progressbar=False, random_seed=7)

    Oo_post = idata.posterior["O_o_true"].mean(dim=["chain", "draw"]).values
    Op_post = idata.posterior["O_p_true"].mean(dim=["chain", "draw"]).values
    sigma_o = dyn.rolling_causal_variance(Oo_post, window=6)
    # Same directional-entropy construction validated 2026-09-20 for the
    # production frac_high_entropy feature (see dynamics.py's own
    # docstring) -- this diagnostic builds on the already-fixed measure,
    # not the superseded symmetric one.
    sigma_p = dyn.rolling_causal_downside_variance(Op_post, window=6)
    regime = dyn.classify_regime(sigma_o, sigma_p)

    n = len(window_years)
    # Same mid-window slice score_institution.py already uses for d_A_trend.
    early = slice(max(0, n // 2 - 3), max(1, n // 2))
    # Same trailing slice the production frac_high_entropy feature already uses.
    late = slice(max(0, n - 5), n)

    frac_early = float(np.mean(regime[early] == "high-entropy"))
    frac_late = float(np.mean(regime[late] == "high-entropy"))
    within_window_trend = frac_late - frac_early

    print(f"\n{name} ({unitid}): real window {start_year}-{end_year - 1} ({n} periods)")
    print(f"  full regime series (period-by-period): {list(regime)}")
    print(f"  frac_high_entropy, EARLY segment (periods {early.start}:{early.stop}) = {frac_early:.4f}")
    print(f"  frac_high_entropy, LATE segment (periods {late.start}:{late.stop}, "
          f"matches the production feature's own window) = {frac_late:.4f}")
    print(f"  within_window_trend (late - early) = {within_window_trend:+.4f}")

    return {
        "unitid": unitid, "name": name, "n_periods": n,
        "frac_early": frac_early, "frac_late": frac_late,
        "within_window_trend": within_window_trend,
    }


def main():
    results = []
    for unitid, name, sector, start_year in TARGETS:
        print(f"\n{'=' * 90}\nComputing within_window_trend: {name} ({unitid})\n{'=' * 90}")
        try:
            r = compute_within_window_trend(unitid, name, sector, start_year)
            if r:
                results.append(r)
        except Exception as e:
            print(f"REAL ERROR computing within_window_trend for {name} ({unitid}): "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for r in results:
        wwt = r["within_window_trend"]
        if wwt < -FLAT_THRESHOLD:
            verdict = "NEGATIVE -- consistent with entropy declining within the window (stabilizing-after-shock read)"
        elif wwt > FLAT_THRESHOLD:
            verdict = "POSITIVE -- entropy still rising within the window (does not support a stabilizing read)"
        else:
            verdict = f"FLAT (|trend| <= {FLAT_THRESHOLD}) -- no meaningful within-window movement either way"
        print(f"  {r['name']:35s} within_window_trend = {wwt:+.4f}  [{verdict}]")

    print(f"\n{'=' * 90}")
    print("READING THIS RESULT:")
    print("Sweet Briar's real, confirmed 2015-16 crisis-and-recovery is the actual test of")
    print("this candidate feature. If Sweet Briar's within_window_trend does NOT come back")
    print("meaningfully negative here, that is real evidence against this specific")
    print("construction -- document it as such, do not force it into the classifier anyway.")
    print("Phoenix's result is exploratory only (see module docstring) -- read alongside")
    print("Sweet Briar's, not as independent confirmation of the same kind.")
    print("This script does not update panel.json, dynamics.py, classifier.py, or")
    print("score_institution.py. Promoting this as a real 9th feature is a separate, later")
    print("step requiring a full live re-fit and leave-one-out re-validation of all 54 panel")
    print("institutions (the same kind of validation recompute_panel_entropy.py already did")
    print("for the directional-entropy fix), not automatic from this result alone.")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
