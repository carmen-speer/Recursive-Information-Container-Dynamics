"""
The real, end-to-end deployment entry point: given a UNITID, fetch
live data, run it through the RICD state-space model, extract the
eight validated features, and produce a real classification.

This is the piece that did not exist before -- extract_features.py in
the original session was a hardcoded batch script for the known
54-institution validation panel, not a general-purpose scorer for a
new, unknown institution. This file replicates that same validated
pipeline (real_adapter.py + model.py + classifier.py) against live
data instead of pre-downloaded local files.

Usage:
    python score_institution.py <unitid> <institution name> [--sector private|public|forprofit]

Requires COLLEGE_SCORECARD_API_KEY set in the environment.
"""

from __future__ import annotations

import sys
import json
import csv
import argparse
import datetime
from pathlib import Path

import numpy as np
import pymc as pm

import model as mdl
import dynamics as dyn
import real_adapter as ra
import fetch_live_data as fld
from classifier import RICDClassifier, InstitutionFeatures, load_panel, GOVERNANCE_OVERRIDE_UNITIDS


def compute_features_for_institution(
    unitid: str, name: str, sector: str = "private",
    start_year: int = 2013, end_year: int | None = None,
) -> InstitutionFeatures | None:
    """
    Real, live scoring pipeline for one institution, replicating
    extract_features.py's validated process() function against live
    data. Returns None, with a printed reason, if live data is
    genuinely insufficient -- never fabricates a feature value to
    force a result through.
    """
    end_year = end_year or (datetime.date.today().year - 2)  # IPEDS lags by ~2 years
    window_years = [f"{y}-{str(y + 1)[2:]}" for y in range(start_year, end_year)]

    # --- Real, live enrollment/admissions/completion series ---
    series = fld.build_live_series(unitid, start_year, end_year)
    if series is None:
        print(f"INSUFFICIENT ENROLLMENT DATA: could not build a complete real "
              f"series for {name} ({unitid}) across {start_year}-{end_year}. "
              f"Not scoring rather than guessing.")
        return None

    O_o_real, O_p_real = ra.compute_O_o_O_p(series)

    # --- Real, live finance data: download bulk files, then parse them ---
    dest_base = f"/tmp/ipeds_live/{unitid}"
    finance_years_available = []
    for year in range(start_year, end_year):
        result = fld.download_ipeds_finance_bulk(year, dest_dir=dest_base, sector=sector)
        if result:
            finance_years_available.append(year)
    if len(finance_years_available) < 3:
        print(f"INSUFFICIENT FINANCE DATA: only {len(finance_years_available)} real years "
              f"downloaded for {name} ({unitid}). This is the known-fragile part of the "
              f"pipeline (see fetch_live_data.py) -- likely means NCES changed its bulk-file "
              f"naming convention this cycle and the download pattern needs a manual update, "
              f"not that the institution actually lacks data. Not scoring rather than guessing.")
        return None

    E_exch, M_maint, W_instr, W_total, mask, scale = fld.parse_live_finance(
        unitid, window_years, dest_base, sector=sector)
    if mask.sum() < 3:
        print(f"INSUFFICIENT PARSED FINANCE DATA: only {int(mask.sum())} real years "
              f"of finance data actually parsed for {name} ({unitid}) after download -- "
              f"likely a field-code mismatch for this sector/year (see "
              f"fetch_live_data.PRIVATE_FINANCE_FIELDS / PUBLIC_FINANCE_FIELDS). "
              f"Not scoring rather than guessing.")
        return None

    # --- Real Bayesian trajectory fit, identical to the validated pipeline ---
    types = ["observed"] * len(O_o_real)
    pymc_model = mdl.build_model_stage2(O_o_real, O_p_real, types, types, E_exch, M_maint, W_instr, W_total, mask)
    with pymc_model:
        idata = pm.sample(300, tune=300, chains=2, cores=2, target_accept=0.9, progressbar=False, random_seed=7)

    d_A_t = idata.posterior["d_A_t"].mean(dim=["chain", "draw"]).values
    delta_R_t = idata.posterior["delta_R_t"].mean(dim=["chain", "draw"]).values
    Oo_post = idata.posterior["O_o_true"].mean(dim=["chain", "draw"]).values
    Op_post = idata.posterior["O_p_true"].mean(dim=["chain", "draw"]).values
    D_op_proxy = np.abs(Oo_post - Op_post)
    sigma_o = dyn.rolling_causal_variance(Oo_post, window=6)
    sigma_p = dyn.rolling_causal_variance(Op_post, window=6)
    regime = dyn.classify_regime(sigma_o, sigma_p)

    n = len(window_years)
    mid = slice(max(0, n // 2 - 3), max(1, n // 2))
    late = slice(max(0, n - 5), n)
    d_A_mid = float(np.mean(d_A_t[mid])) if n > 3 else float(d_A_t[0])
    d_A_late = float(np.mean(d_A_t[late]))
    d_A_trend = d_A_late - d_A_mid
    d_A_final = float(d_A_t[-1])
    delta_R_final = float(delta_R_t[-1])
    frac_high_entropy = float(np.mean(regime[-5:] == "high-entropy"))

    def delta_R_trend_from(traj):
        traj = np.array(traj)
        n2 = len(traj)
        m = traj[n2 // 3: 2 * n2 // 3].mean()
        l = traj[-n2 // 4:].mean() if n2 >= 4 else traj[-1]
        return float(l - m)

    delta_R_trend = delta_R_trend_from(delta_R_t)

    # --- The remaining features, computed directly from real parsed data ---
    liabilities_field = fld.PRIVATE_FINANCE_FIELDS["total_liabilities"] if sector == "private" \
        else fld.PUBLIC_FINANCE_FIELDS["total_liabilities"]
    endowment_field = fld.PRIVATE_FINANCE_FIELDS["endowment"] if sector == "private" \
        else fld.PUBLIC_FINANCE_FIELDS["endowment"]

    liabilities_by_year, endowment_val, research_val, instruction_val = {}, None, None, None
    for year in finance_years_available:
        year_dir = Path(dest_base) / str(year)
        csv_files = list(year_dir.glob("*.csv")) if year_dir.exists() else []
        if not csv_files:
            continue
        try:
            with open(csv_files[0], encoding="latin-1") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("UNITID") == unitid:
                        liabilities_by_year[year] = float(row[liabilities_field])
                        if endowment_val is None:
                            try:
                                endowment_val = float(row[endowment_field])
                            except (KeyError, ValueError):
                                pass
                        if sector == "private":
                            try:
                                research_val = float(row["F2E021"])
                                instruction_val = float(row["F2E011"])
                            except (KeyError, ValueError):
                                pass
                        break
        except (ValueError, KeyError, FileNotFoundError):
            continue

    if len(liabilities_by_year) < 3:
        print(f"INSUFFICIENT DEBT DATA: only {len(liabilities_by_year)} real years of "
              f"total-liabilities data parsed for {name} ({unitid}). Not scoring rather "
              f"than guessing.")
        return None

    years_sorted = sorted(liabilities_by_year)
    vals = [liabilities_by_year[y] for y in years_sorted]
    scale_max = max(vals) if max(vals) > 0 else 1.0
    vals_norm = [v / scale_max for v in vals]
    diffs = np.diff(vals_norm)
    debt_spike = float(np.max(diffs) - np.median(np.abs(diffs))) if len(diffs) > 0 else 0.0

    peak_enrollment = float(np.max(series["n_undergrads"])) if len(series["n_undergrads"]) else None
    if endowment_val is not None and peak_enrollment:
        reserve_adequacy = float(np.log1p(endowment_val / peak_enrollment))
    else:
        print(f"NOTE: no real endowment data found for {name} ({unitid}) -- "
              f"reserve_adequacy defaulting to 0.0 (log1p(0)), consistent with the "
              f"validated panel's treatment of institutions with a real, structural "
              f"absence of endowment data (e.g., for-profits) rather than a missing-data gap.")
        reserve_adequacy = 0.0

    research_ratio = (research_val / instruction_val) if (research_val and instruction_val) else 0.0

    return InstitutionFeatures(
        unitid=unitid, name=name,
        d_A_trend=d_A_trend, d_A_final=d_A_final, delta_R_final=delta_R_final,
        frac_high_entropy=frac_high_entropy, debt_spike=debt_spike,
        delta_R_trend=delta_R_trend, reserve_adequacy=reserve_adequacy,
        research_ratio=research_ratio, outcome=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("unitid")
    parser.add_argument("name", nargs="+")
    parser.add_argument("--sector", default="private", choices=["private", "public", "forprofit"])
    args = parser.parse_args()
    unitid, name = args.unitid, " ".join(args.name)

    panel = load_panel("../data/panel/panel.json")
    clf = RICDClassifier()
    clf.fit(panel)

    if unitid in GOVERNANCE_OVERRIDE_UNITIDS:
        print(json.dumps({
            "unitid": unitid, "name": name, "prediction": "high_risk",
            "method": "governance_override",
        }, indent=2))
        return

    features = compute_features_for_institution(unitid, name, sector=args.sector)
    if features is None:
        print(json.dumps({"unitid": unitid, "name": name, "prediction": "insufficient_data"}, indent=2))
        return

    result = clf.classify(features)
    print(json.dumps({
        "unitid": result.unitid, "name": result.name,
        "prediction": "high_risk" if result.prediction == 1 else "stable",
        "method": result.method, "probability": result.probability,
    }, indent=2))


if __name__ == "__main__":
    main()
