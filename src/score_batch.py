"""
Scores a small, fixed list of real institutions in one run, using the
same validated pipeline as score_institution.py (compute_features_for_institution
+ RICDClassifier + save_live_score). This is the first real test of the
pipeline against genuinely new institutions -- none of the ones below
are in the original 54-institution validation panel, unlike the
University of Michigan smoke test, which is already a known panel
member (a stable, easy case chosen specifically to test the pipeline
plumbing, not the classifier's real generalization).

Chosen deliberately for real diversity, not convenience:
- Sweet Briar College (233718, private): a real, well-documented
  near-closure -- announced closure in 2015, then reopened after an
  alumnae-led rescue -- a genuinely hard, ambiguous real case, not an
  easy one.
- University of Phoenix-Arizona (484613, for-profit): exercises the
  for-profit code path specifically, including the real, structural
  absence of endowment data for for-profit institutions (see
  score_institution.py's own handling of that case).
- West Virginia University (238032, public): swapped in on 2026-09-15
  for Youngstown State University (206695), which came back
  "insufficient_data" for a real, different reason than the for-profit
  field-code bug above -- fetch_live_data.build_live_series() requires
  every year 2013-2023 to have a complete College Scorecard record
  across enrollment, admission rate, completion rate, and tuition, with
  zero tolerance for a single missing field in a single year, and
  Youngstown hit that wall somewhere in its real record (the exact
  field/year was never pinned down). West Virginia University is a
  real, informative choice for the same reason Youngstown was picked
  originally -- not another large, obviously-safe flagship like
  Michigan, but an institution that went through a genuine, widely
  reported financial crisis and program/faculty cuts starting in 2023
  -- while being large and well-established enough that Scorecard
  suppressing a field for small cohort size is far less likely than it
  may have been for Youngstown. This does not guarantee a clean run --
  the exact original failure was never confirmed, so this is a real,
  reasoned choice rather than a proven fix.

Continues to the next institution if one fails or comes back
"insufficient_data" -- one hard institution should never block the
others, and a real, honest "insufficient_data" result is itself
useful information, not a reason to stop.
"""

from score_institution import compute_features_for_institution, save_live_score, prune_stale_live_scores

import json

from classifier import RICDClassifier, load_panel, GOVERNANCE_OVERRIDE_UNITIDS
from score_institution import compute_features_for_institution, save_live_score

INSTITUTIONS = [
    ("233718", "Sweet Briar College", "private"),
    ("484613", "University of Phoenix-Arizona", "forprofit"),
    ("238032", "West Virginia University", "public"),
]


def main():
    panel = load_panel("../data/panel/panel.json")
    clf = RICDClassifier()
    clf.fit(panel)

    results = []
    for unitid, name, sector in INSTITUTIONS:
        print(f"\n{'=' * 70}\nScoring {name} ({unitid}, {sector})\n{'=' * 70}")
        try:
            if unitid in GOVERNANCE_OVERRIDE_UNITIDS:
                result_dict = {
                    "unitid": unitid, "name": name, "prediction": "high_risk",
                    "method": "governance_override",
                }
            else:
                features = compute_features_for_institution(unitid, name, sector=sector)
                if features is None:
                    result_dict = {"unitid": unitid, "name": name, "prediction": "insufficient_data"}
                else:
                    result = clf.classify(features)
                    result_dict = {
                        "unitid": result.unitid, "name": result.name,
                        "prediction": "high_risk" if result.prediction == 1 else "stable",
                        "method": result.method, "probability": result.probability,
                    }
        except Exception as e:
            # A real, honest per-institution failure -- print it and move on
            # to the next institution rather than losing the whole batch.
            print(f"REAL ERROR scoring {name} ({unitid}): {type(e).__name__}: {e}")
            result_dict = {"unitid": unitid, "name": name, "prediction": "error", "error": str(e)}

        print(json.dumps(result_dict, indent=2))
        save_live_score(result_dict)
        results.append(result_dict)

        prune_stale_live_scores({unitid for unitid, _, _ in INSTITUTIONS})print(f"\n{'=' * 70}\nBATCH DONE -- {len(results)} institutions attempted\n{'=' * 70}")


if __name__ == "__main__":
    main()
