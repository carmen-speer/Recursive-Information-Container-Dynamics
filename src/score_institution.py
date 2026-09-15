# UNITID of the Michigan smoke test run separately by rescore.yml's
# own score_institution.py CLI step -- it never appears in
# score_batch.py's INSTITUTIONS list, so a pruning pass keyed only to
# that list would wrongly delete it as "stale" on every run.
MICHIGAN_SMOKE_TEST_UNITID = "170976"


def prune_stale_live_scores(current_unitids: set, path: str = "../docs/data/live_scores.json") -> None:
    """
    Removes any saved live score whose unitid is neither in
    current_unitids nor the Michigan smoke test. Real cleanup for the
    Youngstown State -> West Virginia University swap (and any future
    swap): without this, a retired institution's old "insufficient_data"
    row sits on the public dashboard forever, because save_live_score
    only ever adds or updates entries, never removes them. Only called
    from score_batch.py's own run, right after it knows its own real
    current list, so it never prunes an institution that simply hasn't
    been (re-)scored yet this run.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        existing = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return  # a real, honest corrupt/missing file -- nothing to prune
    if not isinstance(existing, list):
        return
    keep_ids = current_unitids | {MICHIGAN_SMOKE_TEST_UNITID}
    pruned = [r for r in existing if r.get("unitid") in keep_ids]
    removed = [r for r in existing if r.get("unitid") not in keep_ids]
    if removed:
        for r in removed:
            print(f"Pruned stale live score: {r.get('name')} ({r.get('unitid')}) -- no longer tracked.")
        p.write_text(json.dumps(pruned, indent=2))
