"""Train the elimination model, backtest it, and simulate the rest of the live season.

Model: conditional logit. Each elimination week is one "race": every couple still
dancing gets a danger score  x·beta , and P(couple eliminated) = softmax(danger) over
that week's couples. This is the same idea as World-Cup-style predictors that turn
team ratings into match probabilities, then Monte Carlo the bracket. Here the
"bracket" is the remaining weeks of the season.

Writes web/data.js for the UI.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RNG = np.random.default_rng(35)
N_SIMS = 20000
FINALISTS = 5  # couples in the finale for recent seasons
SEASON_WEEKS = 11

FEATURES = [
    "score_z",       # this week's judges' score vs. the field (z-score)
    "form_z",        # average z over previous weeks
    "trend",         # this week minus previous form (improving?)
    "bottom_prev",   # times previously in bottom two/three
    "escapes",       # times survived with a bottom-two judges' score (= hidden fan votes)
    "pro_strength",  # pro's historical results before this season
    "wiki_z",        # Wikipedia pageviews in the week before the show vs. the field
    "wiki_missing",  # 1 when no pageview data (before season 21)
    "social_z",      # YouTube views per official performance video (+ IG/TikTok if given)
    "social_missing",
]


# ---------------------------------------------------------------- features
def load():
    couples = pd.read_csv(DATA / "couples.csv")
    perfs = pd.read_csv(DATA / "performances.csv")
    perfs["result"] = perfs["result"].fillna("")
    perfs["score_pct"] = perfs["score"] / (10 * perfs["n_judges"])
    wiki_f = DATA / "wiki_views.csv"
    wiki = pd.read_csv(wiki_f) if wiki_f.exists() else pd.DataFrame(columns=["season", "week", "celebrity", "wiki_views"])
    return couples, perfs, wiki


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def log_z(s):
    """z-score of log(views) within the week's field; NaN when under 3 couples have data."""
    v = np.log1p(pd.to_numeric(s, errors="coerce"))
    return zscore(v) if v.notna().sum() >= 3 else v * np.nan


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


def build(couples, perfs, wiki, social=None):
    """One row per (season, week, celebrity) with features and label."""
    week = (perfs.groupby(["season", "week", "celebrity"])
            .agg(score_pct=("score_pct", "mean"), theme=("theme", "first"),
                 result=("result", lambda r: " | ".join(sorted(set(r)))))
            .reset_index())
    week["scored"] = week["score_pct"].notna()
    week["eliminated"] = week["result"].str.contains("liminated", case=False).astype(int)
    week["bottom"] = week["result"].str.contains("Bottom", case=False).astype(int)
    pros = pro_strength_table(couples)
    pro_of = couples.set_index(["season", "celebrity"]).pro.str.split().str[:2].str.join(" ")
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
        g["low_score"] = (g["score_pct"] <= g["score_pct"].min() + 1e-9).astype(int)  # baseline only
        g["low2"] = (g["score_pct"].rank(method="min") <= 2).astype(int)
        g["wiki_z"] = log_z(g["wiki_views"])
        g["social_z"] = pd.concat([log_z(g[c]) for c in soc_cols], axis=1).mean(axis=1)
        for f in ["wiki", "social"]:
            g[f + "_missing"] = g[f + "_z"].isna().astype(int)
            g[f + "_z"] = g[f + "_z"].fillna(0)
        g["pro_strength"] = [pros[season].get(pro_of.get((season, c), ""), 0.5) - 0.5 for c in g.celebrity]
        g["n_couples"] = len(g)
        rows.append(g)
    df = pd.concat(rows, ignore_index=True).sort_values(["season", "celebrity", "week"])
    grp = df.groupby(["season", "celebrity"])
    df["form_z"] = grp["score_z"].transform(lambda s: s.shift().expanding().mean()).fillna(0)
    df["trend"] = df["score_z"] - df["form_z"]
    df["bottom_prev"] = grp["bottom"].transform(lambda s: s.shift().fillna(0).cumsum())
    df["escape"] = ((df["low2"] == 1) & (df["eliminated"] == 0) & df["scored"]).astype(int)
    df["escapes"] = df.groupby(["season", "celebrity"])["escape"].transform(lambda s: s.shift().fillna(0).cumsum())
    df["max_week"] = df.groupby("season").week.transform("max")
    return df.reset_index(drop=True)


def elimination_weeks(df):
    """Weeks usable for training: someone eliminated, more than 1 couple, not the finale."""
    g = df.groupby(["season", "week"])
    ok = (g.eliminated.transform("sum") > 0) & (g.celebrity.transform("count") > 2)
    ok &= ~df["result"].str.contains("place|Winner|Runner", case=False)
    ok &= df.groupby(["season", "week"]).result.transform(lambda r: ~r.str.contains("place|Winner|Runner").any())
    return df[ok]


# ---------------------------------------------------------------- conditional logit
def fit(df, l2=0.5):
    X = df[FEATURES].to_numpy(float)
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


def week_probs(g, beta):
    """P(eliminated) per couple in one week; sums to the number eliminated (default 1)."""
    u = g[FEATURES].to_numpy(float) @ beta
    e = np.exp(u - u.max())
    return e / e.sum()


# ---------------------------------------------------------------- backtest
def backtest(df, seasons):
    hist = []
    for s in seasons:
        beta = fit(elimination_weeks(df[df.season < s]))
        test = elimination_weeks(df[df.season == s])
        for w, g in test.groupby("week"):
            p = week_probs(g, beta)
            k = int(g.eliminated.sum())
            order = np.argsort(-p)
            pred = set(g.celebrity.iloc[order[:k]])
            actual = set(g.celebrity[g.eliminated == 1])
            low = set(g.celebrity[g.low_score == 1])
            rank_actual = min(int(np.where(g.celebrity.iloc[order].isin(actual))[0][0]) + 1, 99)
            hist.append({"season": s, "week": int(w), "n": len(g), "k": k,
                         "hit": bool(pred & actual), "top3": rank_actual <= 3,
                         # ties for lowest score: credit split as if picking one at random
                         "baseline": min(1.0, len(low & actual) * k / max(len(low), 1)),
                         "random": k / len(g)})
    return pd.DataFrame(hist)


# ---------------------------------------------------------------- live season
def social_table(season):
    """YouTube median views per official video (history + live) and manual IG/TikTok."""
    parts = []
    hist_f, live_f, manual_f = DATA / "youtube_hist.csv", DATA / "social_live.csv", DATA / "social_manual.csv"
    if hist_f.exists():
        parts.append(pd.read_csv(hist_f))
    live = pd.read_csv(live_f) if live_f.exists() else pd.DataFrame(columns=["celebrity"])
    if manual_f.exists():
        live = live.merge(pd.read_csv(manual_f), on="celebrity", how="outer")
    live["season"] = season
    parts.append(live)
    s = pd.concat(parts, ignore_index=True)
    for c in ["yt_median", "instagram", "tiktok", "youtube_views", "youtube_videos"]:
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
    return s


def simulate(df, beta, season, couples):
    """Monte Carlo the rest of the season, one elimination race per week."""
    cur = df[df.season == season]
    last_week = int(cur.week.max())
    this = cur[cur.week == last_week].set_index("celebrity")
    alive = list(this.index)
    n = len(alive)
    weeks = list(range(last_week, SEASON_WEEKS))  # the week after these is the finale
    # one elimination per week until FINALISTS remain; surplus -> double eliminations late on
    n_elim = [1] * len(weeks)
    surplus = n - FINALISTS - len(weeks)
    for i in range(max(surplus, 0)):
        n_elim[-(i % len(weeks)) - 1] += 1
    for i in range(max(-surplus, 0)):
        n_elim[i] = 0

    past = cur[cur.week < last_week]
    hist = [past[past.celebrity == c].sort_values("week").score_z.to_numpy() for c in alive]
    mu = np.array([h[-3:].mean() if len(h) else 0 for h in hist])
    trend = np.array([(h[-1] - h[0]) / (len(h) - 1) if len(h) > 1 else 0 for h in hist])
    sum0 = np.array([h.sum() for h in hist])
    cnt0 = np.array([len(h) for h in hist], float)
    scored = this["scored"].to_numpy()
    z_now = this["score_z"].to_numpy()
    pro = this["pro_strength"].to_numpy()
    b = dict(zip(FEATURES, beta))
    static = sum(b[f] * this[f].to_numpy() for f in
                 ["pro_strength", "wiki_z", "wiki_missing", "social_z", "social_missing"])
    buzz = (this["social_z"] + this["wiki_z"]).to_numpy() / 2
    SD = 0.55  # typical week-to-week wobble in a couple's z-score

    out_week = np.zeros((n, len(weeks)))
    final = np.zeros(n)
    wins = np.zeros(n)
    for _ in range(N_SIMS):
        live = np.ones(n, bool)
        bottoms = this["bottom_prev"].to_numpy(float).copy()
        escapes = this["escapes"].to_numpy(float).copy()
        s, k_ = sum0.copy(), cnt0.copy()
        recent = mu.copy()
        for wi, w in enumerate(weeks):
            raw = RNG.normal(mu + 0.3 * trend, SD)
            if w == last_week:
                raw = np.where(scored, z_now, raw)
            idx = np.flatnonzero(live)
            z = raw[idx]
            z = (z - z.mean()) / (z.std() or 1)
            form = np.where(k_[idx] > 0, s[idx] / np.maximum(k_[idx], 1), 0)
            u = (b["score_z"] * z + b["form_z"] * form
                 + b["trend"] * (z - form) + b["bottom_prev"] * bottoms[idx]
                 + b["escapes"] * escapes[idx] + static[idx])
            p = np.exp(u - u.max())
            p /= p.sum()
            k = min(n_elim[wi], len(idx) - 1)
            if k:
                gone = RNG.choice(len(idx), size=k, replace=False, p=p)
                p2 = p.copy()
                p2[gone] = -1
                bottoms[idx[int(np.argmax(p2))]] += 1  # next most at risk = "bottom two"
                out_week[idx[gone], wi] += 1
                live[idx[gone]] = False
            low2 = idx[np.argsort(z)[:2]]
            escapes[low2[live[low2]]] += 1
            s[idx] += z
            k_[idx] += 1
            recent[idx] = 0.5 * recent[idx] + 0.5 * z
        fin = np.flatnonzero(live)
        final[fin] += 1
        # finale: no fitted finale data, so rank by fan buzz + recent scoring form
        strength = 0.6 * buzz[fin] + 0.4 * recent[fin] + RNG.gumbel(size=len(fin)) * 0.8
        wins[fin[int(np.argmax(strength))]] += 1

    out_week = {c: {w: out_week[i, wi] for wi, w in enumerate(weeks)} for i, c in enumerate(alive)}
    return weeks, n_elim, out_week, dict(zip(alive, final)), dict(zip(alive, wins))


def main():
    couples, perfs, wiki = load()
    season = int(couples.season.max())
    social = social_table(season)
    df = build(couples, perfs, wiki, social)
    # live model also learns from this season's finished weeks (that's the weekly update)
    train = elimination_weeks(df)
    beta = fit(train)
    print("coefficients (positive = more likely to go home):", file=sys.stderr)
    for f, b in zip(FEATURES, beta):
        print(f"  {f:>13}: {b:+.3f}", file=sys.stderr)

    bt = backtest(df, range(21, season + 1))
    past = bt[bt.season < season]
    summary = {
        "weeks": int(len(past)),
        "model": float(past.hit.mean()),
        "top3": float(past.top3.mean()),
        "lowest_score": float(past.baseline.mean()),
        "random": float(past.random.mean()),
        "seasons": f"{int(past.season.min())}–{int(past.season.max())}",
    }
    print(json.dumps(summary, indent=1), file=sys.stderr)

    # live season: this week's odds + backtest on weeks already played
    cur = df[df.season == season]
    last_week = int(cur.week.max())
    this = cur[cur.week == last_week].copy()
    weeks, n_elim, out_week, final, wins = simulate(df, beta, season, couples)

    cast = couples[couples.season == season].set_index("celebrity")
    soc = social[social.season == season].set_index("celebrity")
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
        couples_out.append({
            "celebrity": c, "pro": info.pro, "couple": info.couple, "notability": info.notability,
            "status": info.status, "place": None if pd.isna(info.place) else int(info.place),
            "alive": c in set(this.celebrity),
            "scores": [{"week": int(r.week), "score": None if pd.isna(r.score) else float(r.score),
                        "max": int(10 * r.n_judges), "dance": r.dance} for r in sc.itertuples()],
            "wiki_views": [{"week": int(r.week), "views": None if pd.isna(r.wiki_views) else float(r.wiki_views)}
                           for r in wiki_cur[wiki_cur.celebrity == c].itertuples()],
            "social": social_of(c),
            "wiki_z": float(this.set_index("celebrity").wiki_z.get(c)) if c in set(this.celebrity) else None,
            "social_z": float(this.set_index("celebrity").social_z.get(c)) if c in set(this.celebrity) else None,
            "form_z": float(h.score_z.mean()) if len(h) else None,
            "pro_strength": float(h.pro_strength.iloc[-1]) + 0.5 if len(h) else None,
            "p_week": {str(w): out_week[c].get(w, 0) / N_SIMS for w in weeks} if c in out_week else {},
            "p_final": final.get(c, 0) / N_SIMS if c in final else 0,
            "p_win": wins.get(c, 0) / N_SIMS if c in wins else 0,
        })

    # most likely elimination order: greedily give each week to its biggest-risk couple
    order, taken = [], set()
    for w, k in zip(weeks, n_elim):
        cands = sorted(((out_week[c].get(w, 0), c) for c in out_week if c not in taken), reverse=True)
        picks = [c for _, c in cands[:k]]
        taken |= set(picks)
        order.append({"week": w, "out": picks, "n": k})
    finalists = sorted((c for c in out_week if c not in taken), key=lambda c: -wins[c])

    themes = cur.groupby("week").theme.first().to_dict()
    past_live = bt[bt.season == season]
    prior_beta = fit(elimination_weeks(df[df.season < season]))
    live_bt = []
    for r in past_live.itertuples():
        g = elimination_weeks(cur)
        g = g[g.week == r.week]
        p = week_probs(g, prior_beta)
        live_bt.append({"week": r.week, "theme": themes.get(r.week, ""), "hit": r.hit,
                        "predicted": list(g.celebrity.iloc[np.argsort(-p)[:max(r.k, 1)]]),
                        "actual": list(g.celebrity[g.eliminated == 1])})

    payload = {
        "season": season, "generated": date.today().isoformat(), "week": last_week,
        "theme": themes.get(last_week, ""), "week_scored": bool(this.scored.all()),
        "themes": {str(k): v for k, v in themes.items()},
        "couples": couples_out, "order": order, "finalists": finalists,
        "backtest": summary, "live_backtest": live_bt,
        "coefficients": dict(zip(FEATURES, map(float, beta))),
        "sims": N_SIMS,
    }
    (ROOT / "web").mkdir(exist_ok=True)
    (ROOT / "web" / "data.js").write_text("window.DWTS = " + json.dumps(payload, indent=1) + ";\n", encoding="utf8")
    risk = sorted(out_week, key=lambda c: -out_week[c][last_week])[:3]
    print(f"wrote web/data.js  (week {last_week}: most at risk = {risk})", file=sys.stderr)


if __name__ == "__main__":
    main()
