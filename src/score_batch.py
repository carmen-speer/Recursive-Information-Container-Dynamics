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
  score_institution.py's own handling of that case), and, as of
  2026-09-15, the real, structural absence of an admission rate in
  every year 2014-2023 (fetch_live_data.build_live_series() now
  tolerates this for sector="forprofit" -- see that function's own
  docstring for why it's safe: nothing in the actual model ever reads
  admission rate). 2013 specifically returned a genuinely empty
  College Scorecard record for this UNITID (confirmed via a real,
  direct per-year diagnostic run, not assumed), so this institution
  starts its window in 2014, one year later than the other two below.
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

Added 2026-09-15, to correct a real selection-bias problem: the three
institutions above were all chosen for reasons connected to their real
or suspected financial condition (a documented near-closure, a
documented enrollment collapse, a documented 2023 financial crisis),
so the live dashboard had drifted to 3 high_risk results against only
1 stable one (the Michigan smoke test) -- not because the model is
biased, but because every institution fed into it so far was picked
*because* it was already publicly known to be distressed. None of
those three results are a novel prediction; all three are public
knowledge already.

The three below were chosen by an explicit, disclosed, neutral rule
applied BEFORE looking at any institution's financial condition, to
avoid cherry-picking "safe" schools just as much as cherry-picking
distressed ones: positions #20, #35, and #48 on CollegeXpress's public
"50 Largest US Colleges and Universities by Enrollment" list
(https://www.collegexpress.com/lists/list/the-50-largest-us-colleges-and-universities/361/),
an ordering by enrollment size that has nothing to do with financial
risk. UNITIDs were then looked up for real from each institution's own
NCES IPEDS profile, not guessed:
- University of Houston (225511, public) -- rank #20.
- California State University-Long Beach (110583, public) -- rank #35.
- Clemson University (217882, public) -- rank #48.
All three are public institutions, so they exercise the
already-validated PUBLIC_FINANCE_FIELDS path, not a new or
recently-patched one.

Real result from that first neutral batch (2026-09-15): all three came
back high_risk. Real reporting supports two of the three -- Cal State
Long Beach has a real, documented enrollment collapse (38,000 to
36,000 students, projected to ~33,000, alongside a real $42M cut) and
Clemson has real, documented rising long-term liabilities ($2.65B, up
$231.9M year-over-year) and expense growth outpacing revenue -- but
University of Houston's real financial condition is the opposite of
high_risk: S&P upgraded its bond rating to AA+ in 2025, citing a
$287M operating surplus and $3.3B in reserves. A real feature-level
diagnostic (src/diagnose_feature_values.py) found the likely cause is
not a data-quality gap but a real, specific feature value: Houston's
frac_high_entropy came back 1.0000 (every one of the last 5 periods
classified into the model's "high-entropy" regime), against exactly
0.0000 for every comparable public flagship already in the validated
panel (Michigan, UVA, UNC, Florida, Wisconsin). Whether that reflects
a genuine regime-detection weakness, an MCMC convergence artifact (the
2-chain/300-draw sampling shows real rhat/ESS problems on every run),
or the classifier failing to distinguish a large POSITIVE resource
shock (the $1.3B Texas University Fund infusion) from a destabilizing
one is not yet resolved -- flagged here rather than smoothed over.

Added 2026-09-15, a second time: three more institutions, to avoid a
different real problem -- after the first neutral batch above came
back 3-for-3 high_risk, adding institutions chosen because they were
*expected* to come back stable would be exactly the same cherry-picking
this whole neutral-selection approach exists to prevent, just aimed in
the opposite direction. So the same neutral rule was extended instead,
with new positions picked purely by position number, decided before
looking up any of the three institutions this pattern would name:
positions #5, #25, and #45 on the same CollegeXpress list (an
arithmetic sequence, 20 ranks apart, starting from position 5 --
avoiding every position already used above). UNITIDs confirmed for
real from each institution's own NCES IPEDS profile page, not guessed:
- University of Central Florida (132903, public) -- rank #5.
- Florida State University (134097, public) -- rank #25.
- University at Buffalo (196088, public) -- rank #45 (not to be
  confused with SUNY Buffalo State, UNITID 196130, a separate
  institution with a similar name).

Removed 2026-09-20: University at Buffalo (196088), taken back out of
this list. After the directional-entropy fix, its live classification
sat right at the decision boundary (48.8% stable, essentially a coin
flip), and feature-level scrutiny found that its real, documented
financial strain -- a $47M federal research-funding cut -- is not a
mechanism this model's eight features are built to detect at all (not
debt, not reserves, not enrollment). Neither its earlier high_risk call
nor this new near-50/50 one can be read as the model having actually
evaluated that real risk, so it's removed rather than left showing an
uninformative number. It was chosen by the same neutral, disclosed rule
as UCF and FSU above and is not being pulled to hide an inconvenient
result -- see the README's Known Gaps section for the full reasoning.

Added 2026-09-22, a third expansion: seven small, tuition-dependent
private liberal arts colleges. This batch had drifted toward large
public flagships and edge cases (Houston, UCF, FSU, Buffalo, Clemson,
Cal State Long Beach) -- exactly the population these eight features
were NOT built to resolve cleanly -- while the population the classifier
was actually validated against (small, tuition-dependent private
colleges) was underrepresented in what the live dashboard shows.

Selected by a disclosed, two-source neutral rule, decided before looking
at any institution's financial condition:
- Source 1: Niche's "Best Liberal Arts Colleges" ranking (216
  institutions, a reputation-based ranking with no relationship to
  financial condition), first 50 positions (the deepest range reliably
  fetchable -- see below), every 7th position: Williams, Vassar,
  Haverford, Bucknell, Thomas Aquinas, Sewanee, Hampden-Sydney.
- Source 2: The Annapolis Group's own published member roster (112
  colleges, ordered by state on the group's own site -- an order with no
  relationship to financial condition), first 50 positions, every 7th
  position: Occidental, Augustana, Wabash, College of the Holy Cross,
  Millsaps, Wittenberg, Bates.
- The two 7-item lists were concatenated (Source 1 first, Source 2
  second) into one 14-item list, and every odd position was kept
  (1, 3, 5, 7, 9, 11, 13): Williams, Haverford, Thomas Aquinas,
  Hampden-Sydney, Augustana, College of the Holy Cross, Wittenberg.
- U.S. News's own liberal-arts ranking, the first-choice source for
  Source 1, blocks automated fetching outright; Niche's ranking was the
  fetchable substitute, and its own pagination did not reliably serve
  results past rank 50 in this environment -- both are real tooling
  limitations, not a choice to stop early once a preferred result
  appeared.
- Williams College is already a confirmed-stable member of the validated
  54-institution panel (UNITID 168342) -- adding it here would be
  redundant with data already scored and validated, not a new test. Per
  a disclosed substitution rule (drop any panel collision, replace with
  the nearest even-numbered position moving forward through the same
  14-item composite list), the next position (2, Vassar College) was
  tried -- also already a confirmed-stable panel member (UNITID 197133)
  -- so the rule moved forward again to position 4 (Bucknell
  University), which is clear of the panel. Two collisions, one
  substitution.

Final seven, UNITIDs confirmed for real from each institution's own NCES
IPEDS Data Feedback Report (not guessed, and not taken from College
Navigator's own page, which did not render institution data through
this project's fetch tooling):
- Bucknell University (211291, Lewisburg, PA) -- substituted in per the
  collision rule above.
- Haverford College (212911, Haverford, PA).
- Thomas Aquinas College (124292, Santa Paula, CA) -- not to be confused
  with St. Thomas Aquinas College, Sparkill, NY (UNITID 195243), a
  separate institution with a similar name.
- Hampden-Sydney College (232256, Hampden-Sydney, VA).
- Augustana College (143084, Rock Island, IL) -- not to be confused with
  Augustana University, Sioux Falls, SD, a separate institution with a
  similar name.
- College of the Holy Cross (166124, Worcester, MA).
- Wittenberg University (206525, Springfield, OH).

All seven are private nonprofit institutions, so this batch exercises
the private-sector finance-parsing path (like Sweet Briar above), not
the already heavily-tested public-sector path most of the batch above
uses. None of their real financial conditions were checked before
selection, and none are checked here either -- that's what live-scoring
them is for.
"""

from __future__ import annotations

import json

from classifier import RICDClassifier, load_panel, GOVERNANCE_OVERRIDE_UNITIDS
from score_institution import compute_features_for_institution, save_live_score, prune_stale_live_scores

INSTITUTIONS = [
    ("233718", "Sweet Briar College", "private", 2013),
    ("484613", "University of Phoenix-Arizona", "forprofit", 2014),
    ("238032", "West Virginia University", "public", 2013),
    ("225511", "University of Houston", "public", 2013),
    ("110583", "California State University-Long Beach", "public", 2013),
    ("217882", "Clemson University", "public", 2013),
    ("132903", "University of Central Florida", "public", 2013),
    ("134097", "Florida State University", "public", 2013),
    ("211291", "Bucknell University", "private", 2013),
    ("212911", "Haverford College", "private", 2013),
    ("124292", "Thomas Aquinas College", "private", 2013),
    ("232256", "Hampden-Sydney College", "private", 2013),
    ("143084", "Augustana College", "private", 2013),
    ("166124", "College of the Holy Cross", "private", 2013),
    ("206525", "Wittenberg University", "private", 2013),
]


def main():
    panel = load_panel("../data/panel/panel.json")
    clf = RICDClassifier()
    clf.fit(panel)

    results = []
    for unitid, name, sector, start_year in INSTITUTIONS:
        print(f"\n{'=' * 70}\nScoring {name} ({unitid}, {sector})\n{'=' * 70}")
        try:
            if unitid in GOVERNANCE_OVERRIDE_UNITIDS:
                result_dict = {
                    "unitid": unitid, "name": name, "prediction": "high_risk",
                    "method": "governance_override",
                }
            else:
                # NUMBA BACKEND (2026-09-22, Thomas Aquinas investigation
                # follow-up): compile_mode="NUMBA" is now the real
                # production setting for the whole batch, not just a
                # diagnostic-only override. See
                # compute_features_for_institution's own docstring in
                # score_institution.py for what this changes, and the
                # README's Known Gaps section for the full evidence
                # (20/20 real trials -- 10 across separate job
                # dispatches, 10 across varied random seeds -- all
                # agreeing on frac_high_entropy under this backend,
                # though real rhat<1.01 convergence was not achieved
                # under any tested backend, Numba included).
                features = compute_features_for_institution(
                    unitid, name, sector=sector, start_year=start_year,
                    cores=1, compile_mode="NUMBA",
                )
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

    # Real cleanup, run once after the whole batch: removes any saved
    # live score for an institution no longer in INSTITUTIONS above
    # (e.g. Youngstown State after the West Virginia University swap,
    # University at Buffalo after its 2026-09-20 removal), so a retired
    # institution doesn't sit on the public dashboard forever as a
    # stale row.
    prune_stale_live_scores({unitid for unitid, _, _, _ in INSTITUTIONS})

    print(f"\n{'=' * 70}\nBATCH DONE -- {len(results)} institutions attempted\n{'=' * 70}")


if __name__ == "__main__":
    main()
