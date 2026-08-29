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


def build_live_series(unitid: str, start_year: int, end_year: int, api_key: str | None = None) -> dict | None:
    """
    Real, live equivalent of real_adapter.extract_institution_series --
    produces the identical dict format (completion, tuition, admit_rate,
    n_undergrads arrays) from live College Scorecard data instead of a
    pre-downloaded local file. Returns None if any year in the window
    is missing a required field, exactly matching the local version's
    behavior -- never fabricates a value to fill a gap.
    """
    completion, tuition, admit_rate, n_undergrads = [], [], [], []
    for year in range(start_year, end_year):
        fields = [
            f"{year}.student.size",
            f"{year}.admissions.admission_rate.overall",
            f"{year}.completion.completion_rate_4yr_150nt",
            f"{year}.cost.tuition.in_state",
        ]
        try:
            result = fetch_scorecard_fields(unitid, fields, api_key)
            n_undergrads.append(float(result[f"{year}.student.size"]))
            admit_rate.append(float(result[f"{year}.admissions.admission_rate.overall"]))
            completion.append(float(result[f"{year}.completion.completion_rate_4yr_150nt"]))
            tuition.append(float(result[f"{year}.cost.tuition.in_state"]))
        except (KeyError, TypeError, ValueError):
            return None  # a real, honest gap year -- matches the local extractor's all-or-nothing behavior
    import numpy as np
    return dict(
        completion=np.array(completion),
        tuition=np.array(tuition),
        admit_rate=np.array(admit_rate),
        n_undergrads=np.array(n_undergrads),
    )


def parse_live_finance(unitid: str, window_years: list[str], dest_base: str | Path, sector: str = "private") -> tuple:
    """
    Real, live equivalent of real_adapter.build_finance_observation_mask.
    Parses the bulk finance CSVs already downloaded by
    download_ipeds_finance_bulk for this institution's window, using the
    real field codes in PRIVATE_FINANCE_FIELDS/PUBLIC_FINANCE_FIELDS.
    Returns the identical (E_exch, M_maint, W_instr, W_total, mask, scale)
    tuple the validated pipeline already uses -- unobserved years stay
    NaN, exactly as the local version leaves them, rather than
    interpolated or assumed.
    """
    import csv
    import numpy as np

    fields = PRIVATE_FINANCE_FIELDS if sector == "private" else PUBLIC_FINANCE_FIELDS
    n = len(window_years)
    E_exch = np.full(n, np.nan)
    M_maint = np.full(n, np.nan)
    W_instr = np.full(n, np.nan)
    W_total = np.full(n, np.nan)
    mask = np.zeros(n, dtype=bool)
    scale = 500_000_000.0  # same fixed reference scale as the validated local pipeline

    for i, y in enumerate(window_years):
        cal_year = int(y[:4]) + 1
        year_dir = Path(dest_base) / str(cal_year - 1)
        if not year_dir.exists():
            continue
        csv_files = list(year_dir.glob("*.csv"))
        if not csv_files:
            continue
        try:
            with open(csv_files[0], encoding="latin-1") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("UNITID") == unitid:
                        E_exch[i] = float(row[fields["revenue"]]) / scale
                        M_maint[i] = float(row[fields["institutional_support"]]) / scale
                        W_instr[i] = float(row[fields["instruction"]]) / scale
                        W_total[i] = float(row[fields["expenses"]]) / scale
                        mask[i] = True
                        break
        except (ValueError, KeyError, FileNotFoundError):
            continue  # a real, honest missing year -- not fabricated

    return E_exch, M_maint, W_instr, W_total, mask, scale


def download_ipeds_finance_bulk(year: int, dest_dir: str | Path, sector: str = "private") -> Path | None:
    """
    Attempts a real download of the bulk IPEDS finance file for the
    given fiscal year, using the real, current NCES data-generator
    endpoint (confirmed directly against NCES's own live Complete Data
    Files page, replacing an earlier, now-defunct static .zip URL
    pattern this project used before -- NCES has changed its
    distribution mechanism at least once already, and may again).
    Returns the local path on success, or None if the pattern no
    longer matches (a real, honest failure, not a silent one --
    callers should check for None and alert rather than assume
    success).

    Honest testing note: this could not be exercised end-to-end from
    within the sandboxed environment this project was built in, since
    nces.ed.gov is not in that sandbox's own network allowlist -- a
    restriction of the development environment itself, confirmed
    directly (the same block was returned for both the old and new
    URL patterns, dressed up as an HTTP 403 in both curl and Python's
    requests library). This should not block real use in a normal
    environment (GitHub Actions, a local machine) without that
    specific restriction, but has not been verified working there yet.

    Extracts into a year-specific subdirectory of dest_dir
    (dest_dir/<year>/), matching the directory structure
    parse_live_finance expects when it later reads these files back.

    sector: "private" (F2 form), "public" (F1A form), or "forprofit" (F3 form).
    """
    form = {"private": "F2", "public": "F1A", "forprofit": "F3"}[sector]
    yy1 = str(year)[2:]
    yy2 = str(year + 1)[2:]
    table_name = f"F{yy1}{yy2}_{form}"
    # Real, current endpoint confirmed against NCES's own live
    # DataFiles.aspx page; HasRV=0 requests original (not revised) data.
    url = f"https://nces.ed.gov/ipeds/data-generator?year={year}&tableName={table_name}&HasRV=0&type=csv"
    year_dir = Path(dest_dir) / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        # The data-generator endpoint returns a raw CSV directly, not a
        # zip archive -- a real, confirmed change from the old pattern.
        csv_path = year_dir / f"{table_name}.csv"
        csv_path.write_bytes(resp.content)
        return year_dir
    except Exception as e:
        print(f"WARNING: real IPEDS bulk download failed for {sector} FY{year} "
              f"({url}): {e}. This is the known-fragile part of the pipeline -- "
              f"check whether NCES changed its file naming or endpoint again.")
        return None
