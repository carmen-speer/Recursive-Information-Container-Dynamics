"""
Bakes docs/data/live_scores.json into docs/index.html as real, static
HTML, so anyone or anything reading the page's raw source -- an AI
summarizer, a search-engine crawler, a screen reader that doesn't run
JavaScript -- sees the actual live-scored results directly, not just
the JavaScript-only placeholder state. The page's own script still
runs for a real browser and re-renders the same data client-side
(harmless and intentionally redundant); this script is what makes the
*pre-JavaScript* source itself already correct.

Run as part of the same automated re-score workflow that updates
docs/data/live_scores.json (see .github/workflows/rescore.yml), so the
static HTML and the JSON it's baked from never drift apart.

Real failure modes handled honestly: if docs/data/live_scores.json is
missing or empty, this bakes in the same "none yet" state the page
already ships with -- it does not fabricate rows.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
INDEX = DOCS / "index.html"
LIVE_SCORES = DOCS / "data" / "live_scores.json"

ROWS_START = "<!-- STATIC_LIVE_ROWS_START -->"
ROWS_END = "<!-- STATIC_LIVE_ROWS_END -->"


def render_row(d: dict) -> str:
    name = d.get("name", "Unknown institution")
    prediction = d.get("prediction", "—")
    pred_class = "closure" if prediction == "high_risk" else ("stable" if prediction == "stable" else "")
    probability = d.get("probability")
    prob_str = f"{probability * 100:.1f}%" if isinstance(probability, (int, float)) else "—"
    method = d.get("method") or "—"
    scored_at = d.get("scored_at")
    if scored_at:
        try:
            scored_str = datetime.fromisoformat(scored_at.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            scored_str = scored_at  # a real, honest fallback if the timestamp format ever changes
    else:
        scored_str = "—"
    return (
        "        <tr>\n"
        f"          <td>{name}</td>\n"
        f"          <td class=\"{pred_class}\">{prediction}</td>\n"
        f"          <td>{prob_str}</td>\n"
        f"          <td>{method}</td>\n"
        f"          <td>{scored_str}</td>\n"
        "        </tr>"
    )


def main() -> None:
    html = INDEX.read_text()
    data = json.loads(LIVE_SCORES.read_text()) if LIVE_SCORES.exists() else []

    start = html.index(ROWS_START) + len(ROWS_START)
    end = html.index(ROWS_END)
    rows_html = ("\n" + "\n".join(render_row(d) for d in data) + "\n") if data else "\n"
    html = html[:start] + rows_html + html[end:]

    def set_display(html: str, marker: str, value: str) -> str:
        idx = html.index(marker)
        style_start = html.index("display:", idx) + len("display:")
        style_end = html.index(";", style_start)
        return html[:style_start] + value + html[style_end:]

    has_data = bool(data)
    html = set_display(html, 'id="live-scope-note"', "block" if has_data else "none")
    html = set_display(html, 'id="live-table"', "table" if has_data else "none")
    html = set_display(html, 'id="live-empty"', "none" if has_data else "block")

    INDEX.write_text(html)
    print(f"Baked {len(data)} live-scored row(s) into {INDEX} (static, pre-JavaScript state).")


if __name__ == "__main__":
    main()
