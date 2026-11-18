"""Draw the README charts (plain SVG, no plotting library) from web/data.js.

Run after model.py:  python docs/make_figures.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
INK, MUTED, GRID, GOLD, RED, TEAL = "#1f1a3a", "#6b6488", "#e6e3ef", "#c99a2e", "#d9485f", "#2a9d8f"
FONT = "font-family='Inter, Segoe UI, Helvetica, Arial, sans-serif'"

LABELS = {
    "judge_pts": "Better judges' rank this week", "judge_pts_save": "Better rank, judges'-save seasons",
    "form_pts": "Better ranks in earlier weeks", "trend": "Improving trend", "bottom_prev": "Times in bottom two",
    "fan_rating": "Higher fan rating", "pro_strength": "Stronger pro partner",
    "cat_athlete": "Athlete (vs actor/TV)", "cat_reality": "Reality star (vs actor/TV)",
    "cat_music": "Musician (vs actor/TV)", "cat_social": "Social media star (vs actor/TV)",
    "social_z": "More YouTube views", "social_missing": "No YouTube data",
}


def load():
    text = (ROOT / "web" / "data.js").read_text(encoding="utf8")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def svg(w, h, body):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' {FONT}>"
            f"<rect width='{w}' height='{h}' fill='white'/>{body}</svg>")


def accuracy(d):
    b = d["backtest"]
    rows = [("Judges + viewers rule", b["model"], GOLD), ("Same inputs, no ranking rule", b["model_logit"], "#b9b4cc"),
            ("Judges' rank only", b["model_judges"], "#b9b4cc"),
            ("Lowest judges' score goes home", b["lowest_score"], "#b9b4cc"), ("Random guess", b["random"], "#b9b4cc")]
    w, h, left, bar_w = 760, 310, 250, 440
    out = [f"<text x='24' y='32' font-size='17' font-weight='600' fill='{INK}'>The model picks who goes home "
           f"{b['model']:.0%} of the time</text>",
           f"<text x='24' y='54' font-size='13' fill='{MUTED}'>Backtest, seasons {b['seasons']} ({b['weeks']} "
           f"elimination weeks), each season predicted from earlier seasons only</text>"]
    for i, (name, v, color) in enumerate(rows):
        y = 84 + i * 40
        out.append(f"<text x='{left - 12}' y='{y + 18}' font-size='14' text-anchor='end' fill='{INK}'>{name}</text>")
        out.append(f"<rect x='{left}' y='{y}' width='{bar_w * v / 0.5:.1f}' height='26' rx='4' fill='{color}'/>")
        out.append(f"<text x='{left + bar_w * v / 0.5 + 8:.1f}' y='{y + 18}' font-size='14' font-weight='600' "
                   f"fill='{INK}'>{v:.0%}</text>")
    out.append(f"<text x='24' y='{h - 12}' font-size='12' fill='{MUTED}'>The eliminated couple was in the model's "
               f"bottom three {b['top3']:.0%} of the time. Log loss {b['log_loss']:.2f} vs {b['log_loss_judges']:.2f} "
               f"(judges only) and {b['log_loss_random']:.2f} (random).</text>")
    (DOCS / "accuracy.svg").write_text(svg(w, h, "".join(out)), encoding="utf8")


def weights(d):
    items = sorted(((LABELS[k], v) for k, v in d["coefficients"].items() if abs(v) >= 0.005), key=lambda kv: kv[1])
    w, row, y0, mid = 760, 26, 76, 430
    scale = min(300, 170 / max(abs(v) for _, v in items))  # longest bar about 170 px
    h = y0 + row * len(items) + 40
    out = [f"<text x='24' y='32' font-size='17' font-weight='600' fill='{INK}'>Low judges' scores and repeat "
           f"bottom-two finishes drive eliminations</text>",
           f"<text x='24' y='54' font-size='13' fill='{MUTED}'>Fan-support model weights. "
           f"<tspan fill='{TEAL}'>Teal = safer</tspan>, <tspan fill='{RED}'>red = more likely to go home</tspan></text>",
           f"<line x1='{mid}' x2='{mid}' y1='{y0 - 8}' y2='{h - 30}' stroke='{MUTED}'/>"]
    for i, (name, v) in enumerate(items):
        y = y0 + i * row
        x, bw = (mid, v * scale) if v > 0 else (mid + v * scale, -v * scale)
        out.append(f"<rect x='{x:.1f}' y='{y}' width='{bw:.1f}' height='18' rx='3' fill='{RED if v > 0 else TEAL}'/>")
        out.append(f"<text x='{mid - 8 + min(v, 0) * scale:.1f}' y='{y + 14}' font-size='13' text-anchor='end' "
                   f"fill='{INK}'>{name}</text>")
        out.append(f"<text x='{mid + 6 + max(v, 0) * scale:.1f}' y='{y + 14}' font-size='12' fill='{MUTED}'>"
                   f"{v:+.2f}</text>")
    (DOCS / "weights.svg").write_text(svg(w, h, "".join(out)), encoding="utf8")


if __name__ == "__main__":
    data = load()
    accuracy(data)
    weights(data)
    print("wrote docs/accuracy.svg and docs/weights.svg")
