"""Train the elimination model, backtest it, and simulate the rest of the live season.

How a week is decided, mirroring the show:

    combined points = judges' rank points + viewers' rank points
    the lowest combined total is in danger

  * Judges' rank points come from the week's judges' scores (best score = most points).
  * Viewers' votes are secret, so their ranking is estimated. Two sources are blended:
      - Instagram likes on @dancingwiththestars' performance posts that week (instagram.py)
      - a fan-support estimate learned from 35 seasons: a conditional logit over form,
        bottom-two history, an Elo-style fan rating, pro partner, celebrity type and
        YouTube views. Its non-judge part says how much support a couple has beyond
        their scores.
  * P(eliminated) = softmax(-k * combined points) over the week's couples, where k is
    fit on past eliminations. The finale applies the same rule repeatedly to rank
    5th, 4th, ... down to the winner.

The season is then simulated 20,000 times, like a World Cup bracket.
Writes web/data.js for the UI and freezes each week's pre-show odds in predictions/.
"""
import json
import re
from functools import lru_cache
from itertools import permutations
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRED = ROOT / "predictions"
RNG = np.random.default_rng(35)
N_SIMS = 20000
TUNE_SEASONS = range(11, 21)   # pick the regularisation strength here...
TEST_SEASONS = range(21, 35)   # ...and report accuracy here, so the test stays untouched
L2_GRID = [0.5, 2.0, 5.0, 10.0, 20.0, 40.0]
FAN_DECAY = 0.85               # older fan-vote evidence counts a little less each week
JUDGE_SCALE = 3.0              # judges-only danger = softmax(-3 * judges' rank points)
LAMBDA_DEFAULT = 0.5           # Instagram's share of the viewer estimate until enough weeks exist to fit it
MIN_IG_WEEKS = 4               # elimination weeks with Instagram likes needed before fitting that share

FEATURES = [
    "judge_pts",      # judges' rank points this week (1 = best score, 0 = worst)
    "judge_pts_save", # same, in judges'-save seasons (judges pick who leaves from the bottom two)
    "form_pts",       # average judges' rank points in earlier weeks
    "trend",          # this week minus earlier form (improving?)
    "bottom_prev",    # times previously in the bottom two/three
    "fan_rating",     # running fan-vote rating: survived when judges said danger (+), bottom two when safe (-)
    "pro_strength",   # pro's historical results before this season
    "cat_athlete",    # celebrity type (actors/TV hosts are the baseline)
    "cat_reality",
    "cat_music",
    "cat_social",
    "social_z",       # YouTube views per official performance video
    "social_missing",
]
# terms that move with this week's judges' scores; everything else is "support beyond the scores"
JUDGE_TERMS = ["judge_pts", "judge_pts_save", "trend"]

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
    seasons = pd.read_csv(DATA / "seasons.csv")
    ig_f = DATA / "instagram_likes.csv"
    ig = pd.read_csv(ig_f) if ig_f.exists() else pd.DataFrame(columns=["season", "week", "celebrity", "posts", "likes"])
    return couples, perfs, seasons, ig


def social_table(season):
    """YouTube median views per official video (history + live) and manual TikTok counts."""
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
    for c in ["yt_median", "tiktok", "youtube_views", "youtube_videos"]:
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


# ---------------------------------------------------------------- rank points
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


def ranks(values):
    """1 = highest value; ties share the better rank (the show's convention)."""
    v = np.asarray(values, float)
    return np.array([1 + np.sum(v > x) for x in v])


def rank_points(values):
    """Rank turned into points from 1 (best) down to 0 (worst), so any field size compares."""
    v = np.asarray(values, float)
    n = len(v)
    return (n - ranks(v)) / (n - 1) if n > 1 else np.ones(n)


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

    After each result, a couple gains what the judges' ranks alone said their danger was
    if they survived, and loses the rest if they landed in the bottom two anyway.
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
            danger = softmax(-JUDGE_SCALE * g.judge_pts.to_numpy())
            hit = ((g.bottom + g.eliminated) > 0).to_numpy()
            for c, d, h in zip(g.celebrity, danger, hit):
                r[c] = FAN_DECAY * r.get(c, 0.0) + (d - 1.0 if h else d)
    return rating


def build(couples, perfs, social, seasons, ig):
    """One row per (season, week, celebrity) with features and label."""
    week = (perfs.groupby(["season", "week", "celebrity"])
            .agg(score_pct=("score_pct", "mean"), score_total=("score", "sum"), theme=("theme", "first"),
                 dance=("dance", lambda d: " & ".join(str(x) for x in d if str(x) != "nan")),
                 result=("result", lambda r: " | ".join(sorted(set(r)))))
            .reset_index())
    week["scored"] = week["score_pct"].notna()
    week["eliminated"] = week["result"].str.contains("liminated", case=False).astype(int)
    week["bottom"] = week["result"].str.contains("Bottom", case=False).astype(int)
    week["result_known"] = week.groupby(["season", "week"]).eliminated.transform("sum") > 0
    # judges'-save era (seasons 28-31): fans pick the bottom two, the judges pick who leaves
    save = set(seasons[seasons.judges_save == 1].season)
    week["judges_save"] = week["season"].isin(save).astype(int)

    pros = pro_strength_table(couples)
    info = couples.set_index(["season", "celebrity"])
    pro_of = info.pro.str.split().str[:2].str.join(" ")
    cat_of = info.notability.map(category)
    week = week.merge(social[["season", "celebrity", "yt_median", "tiktok"]].drop_duplicates(["season", "celebrity"]),
                      on=["season", "celebrity"], how="left")
    week = week.merge(ig[["season", "week", "celebrity", "likes"]], on=["season", "week", "celebrity"], how="left")

    rows = []
    for (season, w), g in week.groupby(["season", "week"]):
        g = g.copy()
        filled = g["score_pct"].fillna(g["score_pct"].mean() if g.scored.any() else 0.5)
        g["score_z"] = zscore(filled)
        g["judge_rank"] = ranks(filled)
        g["judge_pts"] = rank_points(filled) if g.scored.any() else 0.5
        g["judge_pts_save"] = g["judge_pts"] * g["judges_save"]
        g["low_score"] = (g["score_pct"] <= g["score_pct"].min() + 1e-9).astype(int)  # baseline only
        has_ig = g["likes"].notna()
        g["ig_rank"] = np.nan
        g["ig_pts"] = np.nan
        if has_ig.sum() >= 3:  # Instagram ranks only mean something across most of the field
            g.loc[has_ig, "ig_rank"] = ranks(g.loc[has_ig, "likes"])
            g.loc[has_ig, "ig_pts"] = rank_points(g.loc[has_ig, "likes"])
        g["social_z"] = pd.concat([log_z(g[c]) for c in ["yt_median", "tiktok"]], axis=1).mean(axis=1)
        g["social_missing"] = g["social_z"].isna().astype(int)
        g["social_z"] = g["social_z"].fillna(0)
        g["pro_strength"] = [pros[season].get(pro_of.get((season, c), ""), 0.5) - 0.5 for c in g.celebrity]
        g["category"] = [cat_of.get((season, c), "actor/tv") for c in g.celebrity]
        rows.append(g)
    df = pd.concat(rows, ignore_index=True).sort_values(["season", "celebrity", "week"]).reset_index(drop=True)
    for cat in ["athlete", "reality", "music", "social"]:
        df["cat_" + cat] = (df["category"] == cat).astype(int)
    grp = df.groupby(["season", "celebrity"])
    df["form_pts"] = grp["judge_pts"].transform(lambda s: s.shift().expanding().mean()).fillna(0.5)
    df["form_z"] = grp["score_z"].transform(lambda s: s.shift().expanding().mean()).fillna(0)
    df["trend"] = df["judge_pts"] - df["form_pts"]
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


# ---------------------------------------------------------------- conditional logit (fan-support model)
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
    """Plain conditional-logit odds for one week (kept for comparison in the backtest)."""
    return softmax(g[features or FEATURES].to_numpy(float) @ beta)


# ---------------------------------------------------------------- judges + viewers, combined
def fan_support(g, beta):
    """Support beyond this week's scores, from the fan-support model (higher = safer)."""
    b = dict(zip(FEATURES, beta))
    rest = sum(b[f] * g[f].to_numpy(float) for f in FEATURES if f not in JUDGE_TERMS)
    return -rest


def viewer_points(fan, ig_pts, lam):
    """Viewers' rank points: the model's fan estimate blended with Instagram where it exists."""
    fan_pts = rank_points(fan)
    ig = np.asarray(ig_pts, float)
    blend = np.where(np.isnan(ig), fan_pts, (1 - lam) * fan_pts + lam * np.nan_to_num(ig))
    return rank_points(blend), fan_pts


def combined_week(g, beta, kc, lam):
    """Everything about one week under the show's rule. Returns a dict of arrays."""
    judge = g["judge_pts"].to_numpy(float)
    viewer, fan_pts = viewer_points(fan_support(g, beta), g["ig_pts"].to_numpy(float), lam)
    total = judge + viewer
    return {"judge_pts": judge, "viewer_pts": viewer, "fan_pts": fan_pts, "combined": total,
            "p": softmax(-kc * total)}


def fit_kc(train, beta, lam):
    """How sharply a low combined total turns into elimination, fit on past weeks."""
    weeks = [(g, int(g.eliminated.sum())) for _, g in train.groupby(["season", "week"])]
    pre = [(combined_week(g, beta, 1.0, lam)["combined"], g.eliminated.to_numpy(bool)) for g, _ in weeks]

    def nll(kc):
        return -sum(np.log(np.clip(softmax(-kc * c)[y], 1e-9, 1)).sum() for c, y in pre)

    return float(minimize_scalar(nll, bounds=(0.1, 30), method="bounded").x)


def fit_lambda(df, beta, kc):
    """Instagram's share of the viewer estimate, once enough eliminations have likes data."""
    weeks = [g for _, g in elimination_weeks(df).groupby(["season", "week"]) if g.ig_pts.notna().sum() >= 3]
    if len(weeks) < MIN_IG_WEEKS:
        return LAMBDA_DEFAULT, len(weeks), "default"
    grid = np.linspace(0, 1, 21)
    ll = [sum(np.log(np.clip(combined_week(g, beta, kc, lam)["p"][g.eliminated.to_numpy(bool)], 1e-9, 1)).sum()
              for g in weeks) for lam in grid]
    return float(grid[int(np.argmax(ll))]), len(weeks), "fitted"


# ---------------------------------------------------------------- evaluation
def backtest(df, seasons, l2=0.5, combined=True, features=None):
    """Predict each season using only earlier seasons. One row per couple-week.

    combined=True scores the show's rule (judges' + viewers' rank points); False scores the
    plain conditional logit, for comparison.
    """
    out = []
    for s in seasons:
        train = elimination_weeks(df[df.season < s])
        beta = fit(train, l2, features)
        kc = fit_kc(train, beta, 0.0) if combined else None
        for w, g in elimination_weeks(df[df.season == s]).groupby("week"):
            p = combined_week(g, beta, kc, LAMBDA_DEFAULT)["p"] if combined else week_probs(g, beta, features)
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
    """Elimination weeks left before the finale, and how many go home in each."""
    total = seasons.set_index("season").weeks.get(season)
    total = int(total) if pd.notna(total) else 11
    done = df[df.season.between(season - 5, season - 1)]
    final_sizes = done[done.week == done.groupby("season").week.transform("max")].groupby("season").size()
    finalists = int(round(final_sizes.median())) if len(final_sizes) else 5
    weeks = list(range(last_week, total))  # week `total` is the finale
    n_alive = int((df[(df.season == season) & (df.week == last_week)]).shape[0])
    n_elim = [1] * len(weeks)
    if weeks:
        surplus = n_alive - finalists - len(weeks)
        for i in range(max(surplus, 0)):  # more couples than weeks: double eliminations, late first
            n_elim[-(i % len(weeks)) - 1] += 1
        for i in range(max(-surplus, 0)):  # fewer: weeks with no elimination, early first
            n_elim[i] = 0
    return weeks, n_elim, finalists, total


def simulate(df, beta, kc, lam, season, weeks, n_elim, total_weeks, offsets):
    """Monte Carlo the rest of the season: elimination weeks, then the finale ranking."""
    cur = df[df.season == season]
    first = weeks[0] if weeks else total_weeks
    this = cur[cur.week == first].set_index("celebrity")
    alive = list(this.index)
    n = len(alive)
    past = cur[cur.week < first]
    hist_z = [past[past.celebrity == c].sort_values("week").score_z.to_numpy() for c in alive]
    hist_pts = [past[past.celebrity == c].sort_values("week").judge_pts.to_numpy() for c in alive]
    mu = np.array([h[-3:].mean() if len(h) else 0 for h in hist_z])
    trend = np.array([(h[-1] - h[0]) / (len(h) - 1) if len(h) > 1 else 0 for h in hist_z])
    # this week's dance, if announced, nudges the projected score (a rumba scores lower than a jive)
    style_now = np.array([offsets.get(dance_style(d), 0.0) for d in this["dance"]])
    sum0 = np.array([h.sum() for h in hist_pts])
    cnt0 = np.array([len(h) for h in hist_pts], float)
    scored = this["scored"].to_numpy()
    pts_now = this["judge_pts"].to_numpy()
    # Instagram: this week's likes if posted, else the latest week's (fan bases carry over)
    ig_latest = cur[cur.ig_pts.notna()]
    ig_week = int(ig_latest.week.max()) if len(ig_latest) else None
    ig_hold = (ig_latest[ig_latest.week == ig_week].set_index("celebrity").likes.reindex(alive).to_numpy(float)
               if ig_week else np.full(n, np.nan))
    b = dict(zip(FEATURES, beta))
    save = int(this["judges_save"].iloc[0])
    b_judge = b["judge_pts"] + b["judge_pts_save"] * save + b["trend"]
    static = sum(b[f] * this[f].to_numpy() for f in
                 ["pro_strength", "cat_athlete", "cat_reality", "cat_music", "cat_social", "social_z", "social_missing"])
    SD = 0.55  # typical week-to-week wobble in a couple's z-score

    out_week = np.zeros((n, len(weeks)))
    final = np.zeros(n)
    place = np.zeros((n, n + 1))  # place[i, k] = times couple i finished k-th in the finale
    for _ in range(N_SIMS):
        live = np.ones(n, bool)
        bottoms = this["bottom_prev"].to_numpy(float).copy()
        fan = this["fan_rating"].to_numpy(float).copy()
        s, k_ = sum0.copy(), cnt0.copy()

        def week_scores(wi, w, idx):
            raw = RNG.normal(mu + 0.3 * trend + (style_now if wi == 0 else 0), SD)[idx]
            jp = rank_points(raw)
            if w == first and scored.all():
                jp = pts_now[idx]
            form = np.where(k_[idx] > 0, s[idx] / np.maximum(k_[idx], 1), 0.5)
            support = -((b["form_pts"] - b["trend"]) * form + b["bottom_prev"] * bottoms[idx]
                        + b["fan_rating"] * fan[idx] + static[idx])
            ig = np.where(np.isnan(ig_hold[idx]), np.nan, rank_points(np.nan_to_num(ig_hold[idx])))
            if np.isnan(ig_hold[idx]).sum() > len(idx) - 3:
                ig = np.full(len(idx), np.nan)
            viewer, _ = viewer_points(support, ig, lam)
            return jp, jp + viewer

        for wi, w in enumerate(weeks):
            idx = np.flatnonzero(live)
            jp, total = week_scores(wi, w, idx)
            p = softmax(-kc * total)
            k = min(n_elim[wi], len(idx) - 1)
            danger = softmax(-JUDGE_SCALE * jp)
            if k:
                gone = RNG.choice(len(idx), size=k, replace=False, p=p)
                p2 = p.copy()
                p2[gone] = -1
                nxt = int(np.argmax(p2))  # next most at risk = "bottom two"
                hit = np.zeros(len(idx), bool)
                hit[gone] = True
                hit[nxt] = True
                bottoms[idx[nxt]] += 1
                out_week[idx[gone], wi] += 1
                live[idx[gone]] = False
                fan[idx] = FAN_DECAY * fan[idx] + np.where(hit, danger - 1, danger)
            s[idx] += jp
            k_[idx] += 1
        # finale: the same rule ranks everyone, lowest combined total out first
        idx = np.flatnonzero(live)
        final[idx] += 1
        _, total = week_scores(len(weeks), total_weeks, idx)
        left = list(range(len(idx)))
        for pos in range(len(idx), 0, -1):
            q = softmax(-kc * total[left])
            out = left[RNG.choice(len(left), p=q)]
            place[idx[out], pos] += 1
            left.remove(out)

    out_week = {c: {w: out_week[i, wi] / N_SIMS for wi, w in enumerate(weeks)} for i, c in enumerate(alive)}
    p_place = {c: {str(k): place[i, k] / N_SIMS for k in range(1, n + 1) if place[i, k]} for i, c in enumerate(alive)}
    return out_week, dict(zip(alive, final / N_SIMS)), p_place


# ---------------------------------------------------------------- frozen predictions
def freeze(season, week, scores_in, payload):
    """Save this week's odds before the result exists, so the live record can't be fudged.

    '_preshow' is written while no scores are in (before the show), '_scores' after the
    judges' scores but before the result. Once a result exists neither is rewritten.
    """
    PRED.mkdir(exist_ok=True)
    tag = "scores" if scores_in else "preshow"
    f = PRED / f"s{season}_w{week:02d}_{tag}.json"
    f.write_text(json.dumps({"season": season, "week": week, "stage": tag,
                             "frozen_at": datetime.now().isoformat(timespec="minutes"), **payload},
                            indent=1), encoding="utf8")
    return f


def scorecard(df, season):
    """Grade every frozen prediction whose week now has a result."""
    rows = []
    for f in sorted(PRED.glob(f"s{season}_w*.json")):
        p = json.loads(f.read_text(encoding="utf8"))
        g = df[(df.season == season) & (df.week == p["week"])]
        if "p_win" in p:  # a finale prediction: graded on the winner
            winner = g.celebrity[g.result.str.contains("Winner")]
            if winner.empty:
                continue
            ranked = list(p["p_win"])
            rows.append({"week": p["week"], "theme": p.get("theme", ""), "stage": p["stage"], "frozen_at": p["frozen_at"],
                         "predicted": ranked[:1], "actual": list(winner), "hit": ranked[0] == winner.iloc[0],
                         "rank_of_actual": ranked.index(winner.iloc[0]) + 1, "p_actual": p["p_win"][winner.iloc[0]]})
            continue
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


@lru_cache(maxsize=None)
def _perms(n):
    return np.array(list(permutations(range(1, n + 1))), dtype=np.int16)


def vote_bounds(judge_scores, eliminated, max_n=9):
    """Which viewer places are consistent with who went home, under the ranking rule?

    Judges' points: best score gets n, ties share the better rank. Viewers' points are
    some order of 1..n. The lowest combined total goes home, a tie broken by fewer votes.
    Tries every viewer order (n! of them, so only small fields) and keeps the ones that
    reproduce the result. Returns {index: (best place, worst place)} and how many orders fit.
    """
    n = len(judge_scores)
    if n > max_n or not any(eliminated):
        return None, 0
    s = np.asarray(judge_scores, float)
    judge = np.array([n - np.sum(s > x) for x in s])
    V = _perms(n)
    key = (judge + V) * 100 + V  # lexicographic: combined total, then votes
    out = np.asarray(eliminated, bool)
    ok = key[:, out].max(axis=1) < key[:, ~out].min(axis=1)
    if not ok.any():
        return None, 0
    places = n + 1 - V[ok]  # place 1 = most votes
    return {i: (int(places[:, i].min()), int(places[:, i].max())) for i in range(n)}, int(ok.sum())


def replay_weeks(df, season, l2, lam, themes):
    """This season's finished weeks, re-predicted by a model trained on past seasons only.

    Shows the judges' and viewers' ranks behind every call. Looser than the frozen record:
    view counts and likes were read after those weeks aired.
    """
    train = elimination_weeks(df[df.season < season])
    beta = fit(train, l2)
    kc = fit_kc(train, beta, 0.0)
    out = []
    for w, g in elimination_weeks(df[df.season == season]).groupby("week"):
        r = combined_week(g, beta, kc, lam)
        p = r["p"]
        k = int(g.eliminated.sum())
        names = list(g.celebrity)
        ranked = [names[i] for i in np.argsort(-p)]
        actual = list(g.celebrity[g.eliminated == 1])
        bounds, n_orders = vote_bounds(g.score_total.to_numpy(float), g.eliminated.to_numpy(bool))
        table = [{"celebrity": c, "score": None if pd.isna(sc) else float(sc),
                  "judge_rank": int(jr), "ig_likes": None if pd.isna(lk) else float(lk),
                  "ig_rank": None if pd.isna(ir) else int(ir),
                  "viewer_rank": int(vr), "combined_rank": int(cr), "p_out": float(po), "out": bool(e),
                  "vote_range": list(bounds[i]) if bounds else None}
                 for i, (c, sc, jr, lk, ir, vr, cr, po, e) in enumerate(zip(
                     names, g.score_total, g.judge_rank, g.likes, g.ig_rank, ranks(r["viewer_pts"]),
                     ranks(r["combined"]), p, g.eliminated))]
        out.append({"week": int(w), "theme": themes.get(w, ""), "predicted": ranked[:k], "actual": actual,
                    "hit": bool(set(ranked[:k]) & set(actual)),
                    "rank_of_actual": min(ranked.index(c) + 1 for c in actual),
                    "p_actual": float(sum(p[names.index(c)] for c in actual)),
                    "has_ig": bool(g.ig_pts.notna().any()), "vote_orders": n_orders,
                    "table": sorted(table, key=lambda t: t["combined_rank"])})
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
    couples, perfs, seasons, ig = load()
    season = int(couples.season.max())
    social = social_table(season)
    df = build(couples, perfs, social, seasons, ig)

    l2, l2_scores = tune_l2(df)
    summary = summarize(backtest(df, TEST_SEASONS, l2))
    summary["l2"] = l2
    logit = summarize(backtest(df, TEST_SEASONS, l2, combined=False))
    judges = summarize(backtest(df, TEST_SEASONS, l2, combined=False, features=["judge_pts"]))
    summary.update({"model_logit": logit["model"], "log_loss_logit": logit["log_loss"],
                    "model_judges": judges["model"], "log_loss_judges": judges["log_loss"]})
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration"}, indent=1), file=sys.stderr)

    # live model also learns from this season's finished weeks (that's the weekly update)
    train = elimination_weeks(df)
    beta = fit(train, l2)
    kc = fit_kc(train, beta, 0.0)
    lam, ig_weeks, lam_source = fit_lambda(df, beta, kc)
    print(f"k = {kc:.2f}, Instagram share = {lam:.2f} ({lam_source}, {ig_weeks} weeks with likes)", file=sys.stderr)
    for f, b in zip(FEATURES, beta):
        print(f"  {f:>14}: {b:+.3f}", file=sys.stderr)

    cur = df[df.season == season]
    last_week = int(cur.week.max())
    this = cur[cur.week == last_week].copy()
    weeks, n_elim, finalists_n, total_weeks = schedule(df, seasons, season, last_week)
    offsets = dance_offsets(df[df.season < season])
    out_week, final, p_place = simulate(df, beta, kc, lam, season, weeks, n_elim, total_weeks, offsets)
    p_win = {c: v.get("1", 0.0) for c, v in p_place.items()}
    finale_mode = not weeks

    themes = cur.groupby("week").theme.first().to_dict()
    scores_in = bool(this.scored.any())
    if not (this.result_known.any() or this.result.str.contains("Winner").any()):
        frozen = ({"theme": themes.get(last_week, ""), "p_win": dict(sorted(p_win.items(), key=lambda kv: -kv[1]))}
                  if finale_mode else
                  {"theme": themes.get(last_week, ""),
                   "p_out": {c: round(v[last_week], 4) for c, v in sorted(out_week.items(), key=lambda kv: -kv[1][last_week])}})
        freeze(season, last_week, scores_in, frozen)
    record = scorecard(df, season)
    hindsight = replay_weeks(df, season, l2, lam, themes)

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
    ig_cur = ig[ig.season == season]

    def social_of(c):
        if c not in soc.index:
            return {}
        r = soc.loc[c]
        return {k: (None if pd.isna(r.get(k)) else float(r.get(k)))
                for k in ["youtube_views", "youtube_videos", "tiktok"] if k in r}

    couples_out = []
    for c, info in cast.iterrows():
        h = cur[cur.celebrity == c].sort_values("week")
        sc = perfs[(perfs.season == season) & (perfs.celebrity == c)].sort_values("week")
        pro = info.pro.split("  ")[0]
        likes = ig_cur[ig_cur.celebrity == c].sort_values("week")
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
            "instagram": [{"week": int(r.week), "likes": float(r.likes), "posts": int(r.posts)} for r in likes.itertuples()],
            "social": social_of(c),
            "yt_per_day": velocity.get(c),
            "fan_rating": float(now.fan_rating[c]) if c in alive else None,
            "pro_strength": float(h.pro_strength.iloc[-1]) + 0.5 if len(h) else None,
            "p_week": {str(w): out_week[c][w] for w in weeks} if c in out_week else {},
            "p_final": final.get(c, 0.0),
            "p_place": p_place.get(c, {}),
            "p_win": p_win.get(c, 0.0),
        })

    # most likely elimination order: greedily give each week to its biggest-risk couple
    order, taken = [], set()
    for w, k in zip(weeks, n_elim):
        cands = sorted(((out_week[c][w], c) for c in out_week if c not in taken), reverse=True)
        picks = [c for _, c in cands[:k]]
        taken |= set(picks)
        order.append({"week": w, "out": picks, "n": k})
    finalists = sorted((c for c in out_week if c not in taken), key=lambda c: -p_win[c])
    # most likely finishing order: expected place, best first
    exp_place = {c: sum(int(k) * v for k, v in pp.items()) / max(sum(pp.values()), 1e-9)
                 for c, pp in p_place.items() if pp}
    finish = sorted(exp_place, key=exp_place.get)

    payload = {
        "season": season, "generated": date.today().isoformat(), "week": last_week,
        "theme": themes.get(last_week, ""), "week_scored": bool(this.scored.all()), "scores_in": scores_in,
        "finale_mode": finale_mode,
        "themes": {str(k): v for k, v in themes.items()},
        "schedule": {"total_weeks": total_weeks, "finalists": finalists_n, "judges_save": bool(this.judges_save.iloc[0])},
        "couples": couples_out, "order": order, "finalists": finalists, "finish": finish,
        "rule": {"k": kc, "instagram_share": lam, "instagram_share_source": lam_source, "instagram_weeks": ig_weeks},
        "backtest": summary, "record": record, "hindsight": hindsight,
        "l2_scores": {str(k): v for k, v in l2_scores.items()},
        "coefficients": dict(zip(FEATURES, map(float, beta))),
        "sims": N_SIMS,
    }
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "data.js").write_text("window.DWTS = " + json.dumps(payload, indent=1) + ";\n", encoding="utf8")
    top = finish[:3] if finale_mode else sorted(out_week, key=lambda c: -out_week[c][last_week])[:3]
    print(f"wrote web/data.js  ({'finale: predicted top 3' if finale_mode else f'week {last_week}: most at risk'} = {top})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
