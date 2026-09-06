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
  monitored (see the GitHub Actions workflow's
