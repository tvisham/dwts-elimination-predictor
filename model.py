"""Train the elimination model, backtest it, and simulate the rest of the live season.

Model: conditional logit. Each elimination week is one "race": every couple still
dancing gets a danger score  x·beta , and P(couple eliminated) = softmax(danger) over
that week's couples. This is the same idea as World-Cup-style predictors that turn
team ratings into match probabilities, then Monte Carlo the bracket. Here the
"bracket" is the remaining weeks of the season.

Writes web/data.js for the UI and freezes each week's pre-show odds in predictions/.
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRED = ROOT / "predictions"
RNG = np.random.default_rng(35)
N_SIMS = 20000
TUNE_SEASONS = range(11, 21)   # pick the regularisation strength here...
TEST_SEASONS = range(21, 35)   # ...and report accuracy here, so the test stays untouched
L2_GRID = [0.5, 2.0, 5.0, 10.0, 20.0, 40.0]
FAN_DECAY = 0.85               # older fan-vote evidence counts a little less each week
JUDGE_SCALE = 0.9              # judges-only danger = softmax(-0.9 * score z), fit once on all seasons

FEATURES = [
    "score_z",        # this week's judges' score vs. the field (z-score)
    "score_z_save",   # same, in judges'-save seasons (judges pick who leaves from the bottom two)
    "form_z",         # average z over previous weeks
    "trend",          # this week minus previous form (improving?)
    "bottom_prev",    # times previously in the bottom two/three
    "fan_rating",     # running fan-vote rating: survived when judges said danger (+), bottom two when safe (-)
    "pro_strength",   # pro's historical results before this season
    "cat_athlete",    # celebrity type (actors/TV hosts are the baseline)
    "cat_reality",
    "cat_music",
    "cat_social",
    "wiki_z",         # Wikipedia pageviews in the week before the show vs. the field
    "wiki_missing",   # 1 when no pageview data (before season 21)
    "social_z",       # YouTube views per official performance video (+ IG/TikTok if given)
    "social_missing",
]

CATEGORIES = {
    "athlete": r"nfl|nba|mlb|nhl|wnba|olympi|paralympi|gymnast|skater|boxer|ufc|wrestl|athlete|player|"
               r"quarterback|linebacker|receiver|baseman|pitcher|football|basketball|baseball|soccer|tennis|"
               r"golf|racing|indy|nascar|swimmer|sprinter|snowboard|skier|jockey|fighter|champion|softball|rugby",
    "reality": r"bachelor|reality|housewives|summer house|mormon wives|love island|love thy|survivor|"
               r"big brother|vanderpump|jersey shore|kardashian|teen mom|duck dynasty|cast member|"
               r"queer eye|selling sunset|the hills|real world|shark tank|storage wars",
    "music":   r"singer|rapper|musician|band|frontman|songwriter|country|pop star|boy band|girl group|"
               r"guitarist|drummer|dj\b|recording artist|harmony|nsync|backstreet|spice",
    "social":  r"social media|tiktok|youtuber|influencer|content creator|internet personality|instagram",
}


# ---------------------------------------------------------------- data
def load():
    couples = pd.read_csv(DATA / "couples.csv")
    perfs = pd.read_csv(DATA / "performances.csv")
    perfs["result"] = perfs["result"].fillna("")
    perfs["score_pct"] = perfs["score"] / (10 * perfs["n_judges"])
    wiki_f = DATA / "wiki_views.csv"
    wiki = pd.read_csv(wiki_f) if wiki_f.exists() else pd.DataFrame(columns=["season", "week", "celebrity", "wiki_views"])
    seasons = pd.read_csv(DATA / "seasons.csv")
    return couples, perfs, wiki, seasons


def social_table(season):
    """YouTube median views per official video (history + live) and manual IG/TikTok."""
    parts = []
    hist_f, live_f, manual_f = DATA / "youtube_hist.csv", DATA / "social_live.csv", DATA / "social_manual.csv"
    if hist_f.exists():
        parts.append(pd.read_csv(hist_f))
    live = pd.read_csv(live_f) if live_f.exists() else pd.DataFrame(columns=["celebrity"])
    if manual_f.exists():
        live = live.merge(pd.read_csv(manual_f), on="celebrity", how="outer", suffixes=("", "_m"))
    live["season"] = season
    parts.append(live)
    s = pd.concat(parts, ignore_index=True)
    for c in ["yt_median", "instagram", "tiktok", "youtube_views", "youtube_videos"]:
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
    return s


def category(notability):
    """Celebrity type from the cast table's notability line; the first keyword wins.

    "Television actress & singer" -> actor/tv, "Hanson frontman" -> music.
    """
    text = str(notability).lower()
    hits = [(m.start(), cat) for cat, pattern in {**CATEGORIES, "actor/tv": r"actor|actress|host"}.items()
            if (m := re.search(pattern, text))]
    return min(hits)[1] if hits else "actor/tv"


def dance_style(dance):
    """'Paso doble' -> 'paso doble'; fusion or multi-dance weeks -> None."""
    d = str(dance).lower().strip()
    if not d or d == "nan" or "&" in d or "+" in d or "fusion" in d:
        return None
    return re.sub(r"[^a-z ]", "", d.replace("-", " ")).strip() or None


# ---------------------------------------------------------------- features
def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def log_z(s):
    """z-score of log(views) within the week's field; NaN when under 3 couples have data."""
    v = np.log1p(pd.to_numeric(s, errors="coerce"))
    return zscore(v) if v.notna().sum() >= 3 else v * np.nan


def softmax(u):
    e = np.exp(u - np.max(u))
    return e / e.sum()


def pro_strength_table(couples):
    """Mean normalised finish (1 = won, 0 = first out) of each pro's previous seasons."""
    c = couples.dropna(subset=["place"]).copy()
    size = couples.groupby("season").celebrity.count()
    c["finish"] = 1 - (c["place"] - 1) / (c["season"].map(size) - 1)
    c["pro1"] = c["pro"].str.split().str[:2].str.join(" ")
    out = {}
    for season in sorted(couples.season.unique()):
        prev = c[c.season < season]
        stats = prev.groupby("pro1").finish.agg(["sum", "count"])
        # shrink toward 0.5 with a prior worth 2 seasons
        out[season] = ((stats["sum"] + 1.0) / (stats["count"] + 2)).to_dict()
    return out


def fan_ratings(df):
    """Running fan-vote rating per couple, like an Elo rating for the audience.

    After each result, a couple gains what the judges alone said their danger was if
    they survived, and loses the rest if they landed in the bottom two anyway. Survive
    with the lowest score week after week and the rating climbs: that's the fans.
    """
    rating = np.zeros(len(df))
    for _, idx in df.groupby("season").groups.items():
        s = df.loc[idx].sort_values("week")
        r = {}
        for w, g in s.groupby("week"):
            for i, c in zip(g.index, g.celebrity):
                rating[df.index.get_loc(i)] = r.get(c, 0.0)
            if not g.result_known.iloc[0]:
                continue
            danger = softmax(-JUDGE_SCALE * g.score_z.to_numpy())
            hit = ((g.bottom + g.eliminated) > 0).to_numpy()
            for c, d, h in zip(g.celebrity, danger, hit):
                r[c] = FAN_DECAY * r.get(c, 0.0) + (d - 1.0 if h else d)
    return rating


def build(couples, perfs, wiki, social=None, seasons=None):
    """One row per (season, week, celebrity) with features and label."""
    week = (perfs.groupby(["season", "week", "celebrity"])
            .agg(score_pct=("score_pct", "mean"), theme=("theme", "first"),
                 dance=("dance", lambda d: " & ".join(str(x) for x in d if str(x) != "nan")),
                 result=("result", lambda r: " | ".join(sorted(set(r)))))
            .reset_index())
    week["scored"] = week["score_pct"].notna()
    week["eliminated"] = week["result"].str.contains("liminated", case=False).astype(int)
    week["bottom"] = week["result"].str.contains("Bottom", case=False).astype(int)
    week["result_known"] = week.groupby(["season", "week"]).eliminated.transform("sum") > 0
    # judges'-save era (seasons 28-31): fans pick the bottom two, the judges pick who leaves
    save = set(seasons[seasons.judges_save == 1].season) if seasons is not None else set()
    week["judges_save"] = week["season"].isin(save).astype(int)

    pros = pro_strength_table(couples)
    info = couples.set_index(["season", "celebrity"])
    pro_of = info.pro.str.split().str[:2].str.join(" ")
    cat_of = info.notability.map(category)
    week = week.merge(wiki, on=["season", "week", "celebrity"], how="left")
    soc_cols = ["yt_median", "instagram", "tiktok"]
    if social is not None:
        week = week.merge(social[["season", "celebrity"] + soc_cols], on=["season", "celebrity"], how="left")
    else:
        week[soc_cols] = np.nan

    rows = []
    for (season, w), g in week.groupby(["season", "week"]):
        g = g.copy()
        g["score_z"] = zscore(g["score_pct"].fillna(g["score_pct"].mean()))
        g["score_z_save"] = g["score_z"] * g["judges_save"]
        g["low_score"] = (g["score_pct"] <= g["score_pct"].min() + 1e-9).astype(int)  # baseline only
        g["wiki_z"] = log_z(g["wiki_views"])
        g["social_z"] = pd.concat([log_z(g[c]) for c in soc_cols], axis=1).mean(axis=1)
        for f in ["wiki", "social"]:
            g[f + "_missing"] = g[f + "_z"].isna().astype(int)
            g[f + "_z"] = g[f + "_z"].fillna(0)
        g["pro_strength"] = [pros[season].get(pro_of.get((season, c), ""), 0.5) - 0.5 for c in g.celebrity]
        g["category"] = [cat_of.get((season, c), "actor/tv") for c in g.celebrity]
        rows.append(g)
    df = pd.concat(rows, ignore_index=True).sort_values(["season", "celebrity", "week"]).reset_index(drop=True)
    for cat in ["athlete", "reality", "music", "social"]:
        df["cat_" + cat] = (df["category"] == cat).astype(int)
    grp = df.groupby(["season", "celebrity"])
    df["form_z"] = grp["score_z"].transform(lambda s: s.shift().expanding().mean()).fillna(0)
    df["trend"] = df["score_z"] - df["form_z"]
    df["bottom_prev"] = grp["bottom"].transform(lambda s: s.shift().fillna(0).cumsum())
    df["fan_rating"] = fan_ratings(df)
    return df


def elimination_weeks(df):
    """Weeks usable for training: someone eliminated, more than 2 couples, not the finale."""
    g = df.groupby(["season", "week"])
    ok = (g.eliminated.transform("sum") > 0) & (g.celebrity.transform("count") > 2)
    ok &= g.result.transform(lambda r: ~r.str.contains("place|Winner|Runner").any())
    return df[ok]


def dance_offsets(df):
    """How much each dance style moves a couple's score vs. their form (shrunk toward 0)."""
    d = df[(df.week > 1) & df.scored].copy()
    d["style"] = d["dance"].map(dance_style)
    d = d.dropna(subset=["style"])
    d["resid"] = d["score_z"] - d["form_z"]
    stats = d.groupby("style").resid.agg(["sum", "count"])
    return (stats["sum"] / (stats["count"] + 20)).to_dict()


# ---------------------------------------------------------------- conditional logit
def fit(df, l2=0.5, features=None):
    feats = features or FEATURES
    X = df[feats].to_numpy(float)
    y = df["eliminated"].to_numpy(float)
    groups = df.groupby(["season", "week"]).ngroup().to_numpy()
    ng = groups.max() + 1
    n_elim = np.bincount(groups, weights=y, minlength=ng)

    def nll(beta):
        u = X @ beta
        m = np.zeros(ng)
        np.maximum.at(m, groups, u)
        e = np.exp(u - m[groups])
        denom = np.bincount(groups, weights=e, minlength=ng)
        logp = u - m[groups] - np.log(denom[groups])
        p = e / denom[groups]
        loss = -(y * logp).sum() + l2 * beta @ beta
        # gradient: sum over eliminated of (x - E_group[x])
        ex = np.zeros((ng, X.shape[1]))
        np.add.at(ex, groups, p[:, None] * X)
        grad = -(y @ X - (n_elim[:, None] * ex).sum(0)) + 2 * l2 * beta
        return loss, grad

    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return res.x


def week_probs(g, beta, features=None):
    """P(each couple is the one eliminated) in one week; sums to 1."""
    return softmax(g[features or FEATURES].to_numpy(float) @ beta)


# ---------------------------------------------------------------- evaluation
def backtest(df, seasons, l2=0.5, features=None):
    """Predict each season using only earlier seasons. One row per couple-week."""
    out = []
    for s in seasons:
        beta = fit(elimination_weeks(df[df.season < s]), l2, features)
        test = elimination_weeks(df[df.season == s])
        for w, g in test.groupby("week"):
            p = week_probs(g, beta, features)
            k = int(g.eliminated.sum())
            low = g.low_score.to_numpy()
            out.append(pd.DataFrame({
                "season": s, "week": int(w), "celebrity": g.celebrity.to_numpy(), "k": k, "n": len(g),
                "p": np.minimum(k * p, 0.999),  # P(this couple goes home) when k leave
                "rank": (-p).argsort().argsort() + 1,
                "y": g.eliminated.to_numpy(),
                # baseline: lowest judges' score goes home (ties split evenly)
                "p_low": np.minimum(k * low / low.sum(), 0.999),
            }))
    return pd.concat(out, ignore_index=True)


def summarize(bt):
    """Accuracy, log loss, Brier and calibration from couple-week backtest rows."""
    weeks = bt.groupby(["season", "week"])
    # share of the week's eliminated couples the model had in its top k (k = number who left)
    hit = weeks.apply(lambda g: g.sort_values("rank").head(g.k.iloc[0]).y.sum() / g.k.iloc[0])
    top3 = weeks.apply(lambda g: g[g.y == 1]["rank"].min() <= 3)
    low_hit = weeks.apply(lambda g: (g.p_low * g.y).sum() / g.k.iloc[0])
    elim = bt[bt.y == 1]
    eps = 1e-6
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.45, 1.0]
    bt = bt.assign(bin=pd.cut(bt.p, bins, include_lowest=True))
    cal = bt.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return {
        "weeks": int(len(hit)),
        "model": float(hit.mean()), "top3": float(top3.mean()),
        "lowest_score": float(low_hit.mean()),
        "random": float((bt.groupby(["season", "week"]).apply(lambda g: g.k.iloc[0] / g.n.iloc[0])).mean()),
        # log loss on the couples who actually went home (lower is better)
        "log_loss": float(-np.log(elim.p.clip(eps)).mean()),
        "log_loss_random": float(-np.log((elim.k / elim.n).clip(eps)).mean()),
        "brier": float(((bt.p - bt.y) ** 2).mean()),
        "brier_random": float(((bt.k / bt.n - bt.y) ** 2).mean()),
        "calibration": [{"predicted": float(r.predicted), "actual": float(r.actual), "n": int(r.n)}
                        for r in cal.itertuples()],
        "seasons": f"{int(bt.season.min())}–{int(bt.season.max())}",
    }


def tune_l2(df):
    """Choose the ridge penalty on seasons 11–20 by log loss; seasons 21+ stay a clean test."""
    scores = {}
    for l2 in L2_GRID:
        scores[l2] = summarize(backtest(df, TUNE_SEASONS, l2))["log_loss"]
        print(f"  l2={l2:<4} tuning log loss {scores[l2]:.4f}", file=sys.stderr)
    return min(scores, key=scores.get), scores


# ---------------------------------------------------------------- live season
def schedule(df, seasons, season, last_week):
    """Weeks left before the finale and how many go home each week."""
    total = seasons.set_index("season").weeks.get(season)
    total = int(total) if pd.notna(total) else 11
    done = df[df.season.between(season - 5, season - 1)]
    final_sizes = done[done.week == done.groupby("season").week.transform("max")].groupby("season").size()
    finalists = int(round(final_sizes.median())) if len(final_sizes) else 5
    weeks = list(range(last_week, total))  # the last week is the finale
    n_alive = int((df[(df.season == season) & (df.week == last_week)]).shape[0])
    n_elim = [1] * len(weeks)
    surplus = n_alive - finalists - len(weeks)
    for i in range(max(surplus, 0)):  # more couples than weeks: double eliminations, late first
        n_elim[-(i % len(weeks)) - 1] += 1
    for i in range(max(-surplus, 0)):  # fewer: weeks with no elimination, early first
        n_elim[i] = 0
    return weeks, n_elim, finalists, total


def simulate(df, beta, season, weeks, n_elim, offsets):
    """Monte Carlo the rest of the season, one elimination race per week."""
    cur = df[df.season == season]
    last_week = weeks[0]
    this = cur[cur.week == last_week].set_index("celebrity")
    alive = list(this.index)
    n = len(alive)
    past = cur[cur.week < last_week]
    hist = [past[past.celebrity == c].sort_values("week").score_z.to_numpy() for c in alive]
    mu = np.array([h[-3:].mean() if len(h) else 0 for h in hist])
    trend = np.array([(h[-1] - h[0]) / (len(h) - 1) if len(h) > 1 else 0 for h in hist])
    # this week's dance, if announced, nudges the projected score (a rumba scores lower than a jive)
    style_now = np.array([offsets.get(dance_style(d), 0.0) for d in this["dance"]])
    sum0 = np.array([h.sum() for h in hist])
    cnt0 = np.array([len(h) for h in hist], float)
    scored = this["scored"].to_numpy()
    z_now = this["score_z"].to_numpy()
    b = dict(zip(FEATURES, beta))
    save = int(this["judges_save"].iloc[0])
    static = sum(b[f] * this[f].to_numpy() for f in
                 ["pro_strength", "cat_athlete", "cat_reality", "cat_music", "cat_social",
                  "wiki_z", "wiki_missing", "social_z", "social_missing"])
    buzz = (this["social_z"] + this["wiki_z"]).to_numpy() / 2
    SD = 0.55  # typical week-to-week wobble in a couple's z-score

    out_week = np.zeros((n, len(weeks)))
    final = np.zeros(n)
    wins = np.zeros(n)
    for _ in range(N_SIMS):
        live = np.ones(n, bool)
        bottoms = this["bottom_prev"].to_numpy(float).copy()
        fan = this["fan_rating"].to_numpy(float).copy()
        s, k_ = sum0.copy(), cnt0.copy()
        recent = mu.copy()
        for wi, w in enumerate(weeks):
            raw = RNG.normal(mu + 0.3 * trend + (style_now if wi == 0 else 0), SD)
            if w == last_week:
                raw = np.where(scored, z_now, raw)
            idx = np.flatnonzero(live)
            z = raw[idx]
            z = (z - z.mean()) / (z.std() or 1)
            form = np.where(k_[idx] > 0, s[idx] / np.maximum(k_[idx], 1), 0)
            u = ((b["score_z"] + b["score_z_save"] * save) * z + b["form_z"] * form
                 + b["trend"] * (z - form) + b["bottom_prev"] * bottoms[idx]
                 + b["fan_rating"] * fan[idx] + static[idx])
            p = softmax(u)
            k = min(n_elim[wi], len(idx) - 1)
            danger = softmax(-JUDGE_SCALE * z)
            hit = np.zeros(len(idx), bool)
            if k:
                gone = RNG.choice(len(idx), size=k, replace=False, p=p)
                p2 = p.copy()
                p2[gone] = -1
                nxt = int(np.argmax(p2))  # next most at risk = "bottom two"
                bottoms[idx[nxt]] += 1
                hit[gone] = True
                hit[nxt] = True
                out_week[idx[gone], wi] += 1
                live[idx[gone]] = False
                fan[idx] = FAN_DECAY * fan[idx] + np.where(hit, danger - 1, danger)
            s[idx] += z
            k_[idx] += 1
            recent[idx] = 0.5 * recent[idx] + 0.5 * z
        fin = np.flatnonzero(live)
        final[fin] += 1
        # finale: no fitted finale data, so rank by fan buzz + recent scoring form
        strength = 0.6 * buzz[fin] + 0.4 * recent[fin] + RNG.gumbel(size=len(fin)) * 0.8
        wins[fin[int(np.argmax(strength))]] += 1

    out_week = {c: {w: out_week[i, wi] / N_SIMS for wi, w in enumerate(weeks)} for i, c in enumerate(alive)}
    return out_week, dict(zip(alive, final / N_SIMS)), dict(zip(alive, wins / N_SIMS))


# ---------------------------------------------------------------- frozen predictions
def freeze(season, week, scores_in, out_week, theme):
    """Save this week's odds before the result exists, so the live record can't be fudged.

    Two files per week: '_preshow' (written only while no scores are in, i.e. before the
    show) and '_scores' (after the judges' scores, before the result). Once a result
    exists for that week neither is ever rewritten.
    """
    PRED.mkdir(exist_ok=True)
    tag = "scores" if scores_in else "preshow"
    f = PRED / f"s{season}_w{week:02d}_{tag}.json"
    f.write_text(json.dumps({
        "season": season, "week": week, "theme": theme, "stage": tag,
        "frozen_at": datetime.now().isoformat(timespec="minutes"),
        "p_out": {c: round(v[week], 4) for c, v in sorted(out_week.items(), key=lambda kv: -kv[1][week])},
    }, indent=1), encoding="utf8")
    return f


def scorecard(df, season):
    """Grade every frozen prediction whose week now has a result."""
    rows = []
    for f in sorted(PRED.glob(f"s{season}_w*.json")):
        p = json.loads(f.read_text(encoding="utf8"))
        g = df[(df.season == season) & (df.week == p["week"])]
        if not g.result_known.any():
            continue
        out = set(g.celebrity[g.eliminated == 1])
        ranked = list(p["p_out"])
        k = len(out)
        rows.append({
            "week": p["week"], "theme": p.get("theme", ""), "stage": p["stage"], "frozen_at": p["frozen_at"],
            "predicted": ranked[:k], "actual": sorted(out), "hit": bool(set(ranked[:k]) & out),
            "rank_of_actual": min(ranked.index(c) + 1 for c in out if c in ranked) if out & set(ranked) else None,
            "p_actual": float(sum(p["p_out"].get(c, 0) for c in out)),
        })
    return rows


def replay_weeks(df, season, l2, themes):
    """Weeks already played this season, re-predicted by a model trained on past seasons only.

    Not as strict as the frozen record: it uses today's view counts, which already
    include some of the views each couple got after those weeks.
    """
    beta = fit(elimination_weeks(df[df.season < season]), l2)
    out = []
    for w, g in elimination_weeks(df[df.season == season]).groupby("week"):
        p = week_probs(g, beta)
        k = int(g.eliminated.sum())
        ranked = list(g.celebrity.iloc[np.argsort(-p)])
        actual = list(g.celebrity[g.eliminated == 1])
        out.append({"week": int(w), "theme": themes.get(w, ""), "predicted": ranked[:k], "actual": actual,
                    "hit": bool(set(ranked[:k]) & set(actual)),
                    "rank_of_actual": min(ranked.index(c) + 1 for c in actual),
                    "p_actual": float(sum(p[list(g.celebrity).index(c)] for c in actual))})
    return out


def youtube_velocity(season):
    """Views per day on each couple's newest video, from the snapshot log (needs 2+ snapshots)."""
    f = DATA / "snapshots" / "youtube_videos.csv"
    if not f.exists():
        return {}
    v = pd.read_csv(f)
    v = v[v.season == season]
    v["t"] = pd.to_datetime(v.taken_at)
    out = {}
    for c, g in v.groupby("celebrity"):
        first = g.sort_values("t").groupby("video_id").first()
        newest = first.t.idxmax()  # the video we started seeing most recently
        h = g[g.video_id == newest].sort_values("t")
        days = (h.t.iloc[-1] - h.t.iloc[0]).total_seconds() / 86400
        if days >= 0.5:
            out[c] = float((h.views.iloc[-1] - h.views.iloc[0]) / days)
    return out


# ---------------------------------------------------------------- main
def main():
    couples, perfs, wiki, seasons = load()
    season = int(couples.season.max())
    social = social_table(season)
    df = build(couples, perfs, wiki, social, seasons)

    l2, l2_scores = tune_l2(df)
    bt = backtest(df, TEST_SEASONS, l2)
    summary = summarize(bt)
    summary["l2"] = l2
    judges_only = summarize(backtest(df, TEST_SEASONS, l2, ["score_z"]))
    summary["log_loss_judges"] = judges_only["log_loss"]
    summary["model_judges"] = judges_only["model"]
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration"}, indent=1), file=sys.stderr)

    # live model also learns from this season's finished weeks (that's the weekly update)
    beta = fit(elimination_weeks(df), l2)
    print("coefficients (positive = more likely to go home):", file=sys.stderr)
    for f, b in zip(FEATURES, beta):
        print(f"  {f:>14}: {b:+.3f}", file=sys.stderr)

    cur = df[df.season == season]
    last_week = int(cur.week.max())
    this = cur[cur.week == last_week].copy()
    weeks, n_elim, finalists_n, total_weeks = schedule(df, seasons, season, last_week)
    offsets = dance_offsets(df[df.season < season])
    out_week, final, wins = simulate(df, beta, season, weeks, n_elim, offsets)

    themes = cur.groupby("week").theme.first().to_dict()
    scores_in = bool(this.scored.any())
    if not this.result_known.any():
        freeze(season, last_week, scores_in, out_week, themes.get(last_week, ""))
    record = scorecard(df, season)
    hindsight = replay_weeks(df, season, l2, themes)

    photos_f = DATA / "photos.csv"
    photos = pd.read_csv(photos_f) if photos_f.exists() else pd.DataFrame()
    def photo(name, role):
        if photos.empty:
            return None
        hit = photos[(photos.person == name) & (photos.role == role)]
        if hit.empty:
            return None
        r = hit.iloc[0]
        return {"file": r.file, "article": "" if pd.isna(r.article) else r.article,
                "image": "" if pd.isna(r.image) else r.image}
    velocity = youtube_velocity(season)
    cast = couples[couples.season == season].set_index("celebrity")
    soc = social[social.season == season].drop_duplicates("celebrity").set_index("celebrity")
    alive = set(this.celebrity)
    now = this.set_index("celebrity")

    def social_of(c):
        if c not in soc.index:
            return {}
        r = soc.loc[c]
        return {k: (None if pd.isna(r.get(k)) else float(r.get(k)))
                for k in ["youtube_views", "youtube_videos", "instagram", "tiktok"] if k in r}

    wiki_cur = wiki[wiki.season == season]
    couples_out = []
    for c, info in cast.iterrows():
        h = cur[cur.celebrity == c].sort_values("week")
        sc = perfs[(perfs.season == season) & (perfs.celebrity == c)].sort_values("week")
        pro = info.pro.split("  ")[0]
        couples_out.append({
            "celebrity": c, "pro": pro, "couple": info.couple, "notability": info.notability,
            "category": category(info.notability),
            "status": info.status, "place": None if pd.isna(info.place) else int(info.place),
            "alive": c in alive,
            "photo": photo(c, "celebrity"), "pro_photo": photo(pro, "pro"),
            "couple_photo": photo(c, "couple"),
            "scores": [{"week": int(r.week), "score": None if pd.isna(r.score) else float(r.score),
                        "max": int(10 * r.n_judges), "dance": None if pd.isna(r.dance) else r.dance}
                       for r in sc.itertuples()],
            "wiki_views": [{"week": int(r.week), "views": None if pd.isna(r.wiki_views) else float(r.wiki_views)}
                           for r in wiki_cur[wiki_cur.celebrity == c].itertuples()],
            "social": social_of(c),
            "yt_per_day": velocity.get(c),
            "wiki_z": float(now.wiki_z[c]) if c in alive else None,
            "social_z": float(now.social_z[c]) if c in alive else None,
            "fan_rating": float(now.fan_rating[c]) if c in alive else None,
            "form_z": float(h.score_z.mean()) if len(h) else None,
            "pro_strength": float(h.pro_strength.iloc[-1]) + 0.5 if len(h) else None,
            "p_week": {str(w): out_week[c][w] for w in weeks} if c in out_week else {},
            "p_final": final.get(c, 0.0),
            "p_win": wins.get(c, 0.0),
        })

    # most likely elimination order: greedily give each week to its biggest-risk couple
    order, taken = [], set()
    for w, k in zip(weeks, n_elim):
        cands = sorted(((out_week[c][w], c) for c in out_week if c not in taken), reverse=True)
        picks = [c for _, c in cands[:k]]
        taken |= set(picks)
        order.append({"week": w, "out": picks, "n": k})
    finalists = sorted((c for c in out_week if c not in taken), key=lambda c: -wins[c])

    payload = {
        "season": season, "generated": date.today().isoformat(), "week": last_week,
        "theme": themes.get(last_week, ""), "week_scored": bool(this.scored.all()), "scores_in": scores_in,
        "themes": {str(k): v for k, v in themes.items()},
        "schedule": {"total_weeks": total_weeks, "finalists": finalists_n, "judges_save": bool(this.judges_save.iloc[0])},
        "couples": couples_out, "order": order, "finalists": finalists,
        "backtest": summary, "record": record, "hindsight": hindsight, "l2_scores": {str(k): v for k, v in l2_scores.items()},
        "coefficients": dict(zip(FEATURES, map(float, beta))),
        "sims": N_SIMS,
    }
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "data.js").write_text("window.DWTS = " + json.dumps(payload, indent=1) + ";\n", encoding="utf8")
    risk = sorted(out_week, key=lambda c: -out_week[c][last_week])[:3]
    print(f"wrote web/data.js  (week {last_week}: most at risk = {risk})", file=sys.stderr)


if __name__ == "__main__":
    main()
