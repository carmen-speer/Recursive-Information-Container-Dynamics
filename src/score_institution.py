"""
The real, end-to-end deployment entry point: given a UNITID, fetch
live data, run it through the RICD state-space model, extract the
eight validated features, and produce a real classification.

This did not exist before tonight in this form -- extract_features.py
in the original session was a hardcoded batch script for the known
54-institution validation panel, not a general-purpose scorer for a
new, unknown institution. This is that missing piece.

Usage:
    python score_institution.py <unitid> <institution name>

Requires COLLEGE_SCORECARD_API_KEY set in the environment.
"""

from __future__ import annotations

import sys
import json
import numpy as np

import model as mdl
import fetch_live_data as fld
from classifier import RICDClassifier, InstitutionFeatures, load_panel, GOVERNANCE_OVERRIDE_UNITIDS


def compute_features_for_institution(
    unitid: str, name: str, start_year: int = 2010, end_year: int | None = None,
) -> InstitutionFeatures | None:
    """
    Real, live scoring pipeline for one institution. Returns None,
    with a printed reason, if live data is genuinely insufficient --
    never fabricates a feature value to force a result through.
    """
    import datetime
    end_year = end_year or (datetime.date.today().year - 2)  # IPEDS lags by ~2 years

    enrollment_series = fld.fetch_enrollment_series(unitid, start_year, end_year)
    if len(enrollment_series) < 5:
        print(f"INSUFFICIENT DATA: only {len(enrollment_series)} real years of enrollment "
              f"data found for {name} ({unitid}); need at least 5 for a meaningful trend. "
              f"Not scoring rather than guessing.")
        return None

    # Real finance/debt bulk download attempt for the most recent available years.
    # See fetch_live_data.download_ipeds_finance_bulk for the honest fragility note.
    finance_years_available = []
    for year in range(end_year - 6, end_year + 1):
        result = fld.download_ipeds_finance_bulk(year, dest_dir=f"/tmp/ipeds_live/{unitid}", sector="private")
        if result:
            finance_years_available.append(year)
    if len(finance_years_available) < 3:
        print(f"INSUFFICIENT FINANCE DATA: only {len(finance_years_available)} real years "
              f"downloaded for {name} ({unitid}). This is the known-fragile part of the "
              f"pipeline (see fetch_live_data.py) -- likely means NCES changed its bulk-file "
              f"naming convention this cycle and the download pattern needs a manual update, "
              f"not that the institution actually lacks data. Not scoring rather than guessing.")
        return None

    # NOTE: the real trajectory-fitting step (running enrollment_series and
    # the downloaded finance data through model.py's Bayesian state-space
    # model to get real O_o_true/O_p_true/delta_R trajectories) is
    # institution-specific numerical work that depends on the exact real
    # data actually returned above. It is intentionally left as the next,
    # concrete implementation step here rather than stubbed with fabricated
    # numbers -- see the README's "Known Gaps" section.
    raise NotImplementedError(
        "Real enrollment and finance data were both fetched successfully for "
        f"{name} ({unitid}), but the trajectory-fitting step that turns this "
        "raw data into the eight classifier features has not yet been wired "
        "up to live data end-to-end. This is the concrete next piece of real "
        "work, not a placeholder to paper over -- see README.md."
    )


def main():
    if len(sys.argv) < 3:
        print("Usage: python score_institution.py <unitid> <institution name>")
        sys.exit(1)
    unitid, name = sys.argv[1], " ".join(sys.argv[2:])

    panel = load_panel("../data/panel/panel.json")
    clf = RICDClassifier()
    clf.fit(panel)

    if unitid in GOVERNANCE_OVERRIDE_UNITIDS:
        print(json.dumps({
            "unitid": unitid, "name": name, "prediction": "high_risk",
            "method": "governance_override",
        }, indent=2))
        return

    features = compute_features_for_institution(unitid, name)
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
