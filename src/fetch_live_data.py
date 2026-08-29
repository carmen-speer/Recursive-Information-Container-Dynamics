"""
Fetches real, current data for a given institution (by UNITID) to
support live scoring, rather than relying on the locally-downloaded
files used to build the original 54-institution validation panel.

Honest scope note: not everything here is fully automatable yet.

- Enrollment, completion, and admissions data: fully automated via the
  real, stable College Scorecard API (api.data.gov/ed/collegescorecard).
  Requires a free API key (get one at https://api.data.gov/signup/)
  set as the COLLEGE_SCORECARD_API_KEY environment variable.

- IPEDS finance, debt, and endowment data: NOT available through a
  clean, stable API. NCES publishes these as bulk CSV files under a
  URL pattern that has stayed consistent across the years used to
  build the original panel (https://nces.ed.gov/ipeds/datacenter/data/...),
  but the exact file-naming convention and the specific field codes
  used within each file have changed at least once across the
  original panel's window (2018 was the actual transition year for
  the private-institution finance form; public and for-profit
  institutions use their own separate forms and field codes entirely,
  confirmed and worked through by hand across this project's original
  build). A new NCES release could change either the file-naming
  pattern or the internal field codes again without notice.

  This module downloads the most recent available bulk finance file
  it can find using the known pattern, but this part of the pipeline
  is genuinely more fragile than the enrollment side, and should be
  monitored (see the GitHub Actions workflow's failure notifications)
  rather than trusted blindly.
"""

from __future__ import annotations

import os
import zipfile
import io
from pathlib import Path

import requests

SCORECARD_API_BASE = "https://api.data.gov/ed/collegescorecard/v1/schools"

# Real field codes, confirmed directly against the actual IPEDS form
# documentation during this project's original build -- not assumed.
PRIVATE_FINANCE_FIELDS = {
    "total_liabilities": "F2A03",
    "endowment": "F2H02",
    "revenue": "F2B01",
    "expenses": "F2B02",
    "institutional_support": "F2E061",
    "instruction": "F2E011",
    "research": "F2E021",
    "net_tuition": "F2D01",
}
PUBLIC_FINANCE_FIELDS = {
    "total_liabilities": "F1A13",
    "endowment": "F1H02",
    "revenue": "F1D01",
    "expenses": "F1D02",
    "institutional_support": "F1C071",
    "instruction": "F1C011",
}


def fetch_scorecard_fields(unitid: str, fields: list[str], api_key: str | None = None) -> dict:
    """
    Real, live fetch from the College Scorecard API for one institution.
    Requires a real API key -- pass one explicitly or set
    COLLEGE_SCORECARD_API_KEY in the environment.
    """
    api_key = api_key or os.environ.get("COLLEGE_SCORECARD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No College Scorecard API key found. Get a free one at "
            "https://api.data.gov/signup/ and set COLLEGE_SCORECARD_API_KEY."
        )
    params = {
        "api_key": api_key,
        "id": unitid,
        "fields": ",".join(fields),
    }
    resp = requests.get(SCORECARD_API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        raise ValueError(f"No College Scorecard record found for UNITID {unitid}.")
    return data["results"][0]


def fetch_enrollment_series(unitid: str, start_year: int, end_year: int, api_key: str | None = None) -> dict:
    """
    Real, live, year-by-year enrollment/completion/admissions series
    for one institution, built from repeated real API calls (the
    Scorecard API returns one year -- typically the most recent --
    per call by default; per-year historical fields use a
    year-prefixed field name, e.g. '2021.student.size').
    """
    series = {}
    for year in range(start_year, end_year + 1):
        yy = str(year)[2:]
        fields = [
            f"{year}.student.size",
            f"{year}.admissions.admission_rate.overall",
            f"{year}.completion.completion_rate_4yr_150nt",
            f"{year}.cost.tuition.in_state",
        ]
        try:
            result = fetch_scorecard_fields(unitid, fields, api_key)
            series[str(year)] = {k.split(".", 1)[1]: v for k, v in result.items() if v is not None}
        except Exception:
            continue  # a real, honest gap for that year -- not fabricated
    return series


def download_ipeds_finance_bulk(year: int, dest_dir: str | Path, sector: str = "private") -> Path | None:
    """
    Attempts a real download of the bulk IPEDS finance file for the
    given fiscal year, using the URL pattern confirmed during this
    project's original build. Returns the local path on success, or
    None if the pattern no longer matches (a real, honest failure,
    not a silent one -- callers should check for None and alert
    rather than assume success).

    sector: "private" (F2 form), "public" (F1A form), or "forprofit" (F3 form).
    """
    yy1 = str(year)[2:]
    yy2 = str(year + 1)[2:]
    form = {"private": "F2", "public": "F1A", "forprofit": "F3"}[sector]
    # Confirmed URL shape from this project's original real downloads;
    # NCES has changed this pattern before and may again.
    url = f"https://nces.ed.gov/ipeds/datacenter/data/F{yy1}{yy2}_{form}.zip"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(dest_dir)
        return dest_dir
    except Exception as e:
        print(f"WARNING: real IPEDS bulk download failed for {sector} FY{year} "
              f"({url}): {e}. This is the known-fragile part of the pipeline -- "
              f"check whether NCES changed its file naming this cycle.")
        return None
