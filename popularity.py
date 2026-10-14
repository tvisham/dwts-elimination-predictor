"""Collect popularity ("views") signals for each celebrity.

  Wikipedia pageviews  - daily, every celebrity from season 21 on (API starts July 2015).
                         Used both for training and for the live season.
  YouTube              - views on the official DWTS channel's performance videos for the
                         live season (scraped from search results, no API key needed).
  Instagram / TikTok   - those platforms block anonymous scraping, so fill in
                         data/social_manual.csv by hand (follower counts); blank is fine.

Outputs
  data/week_dates.csv   season, week, date        (approximate air date of each week)
  data/wiki_views.csv   season, week, celebrity, wiki_views   (7 days before air date)
  data/social_live.csv  celebrity, youtube_views, youtube_videos, instagram, tiktok
"""
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote, quote_plus

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
SNAP = DATA / "snapshots"
UA = {"User-Agent": "dwts-predictor/0.1 (personal hobby project)"}
BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
FIRST_PAGEVIEW_SEASON = 21
YT_HISTORY_FROM = 28  # official channel posts every performance from here on
MONTHS = "January February March April May June July August September October November December".split()


def parse_date(text):
    m = re.search(r"(%s) (\d{1,2}), (\d{4})" % "|".join(MONTHS), text or "")
    return date(int(m.group(3)), MONTHS.index(m.group(1)) + 1, int(m.group(2))) if m else None


def week_dates(couples, perfs):
    """Air date per (season, week), anchored on the elimination dates in the cast table."""
    last_week = perfs.groupby(["season", "celebrity"]).week.max()
    rows = []
    for season, cast in couples.groupby("season"):
        anchors = []
        for _, c in cast.iterrows():
            d = parse_date(c["status"])
            w = last_week.get((season, c["celebrity"]))
            if d and w and "liminated" in str(c["status"]):
                anchors.append(d - timedelta(days=7 * (int(w) - 1)))
        if not anchors:
            continue
        week1 = sorted(anchors)[len(anchors) // 2]  # median is robust to two-night weeks
        for w in sorted(perfs[perfs.season == season].week.unique()):
            rows.append({"season": season, "week": int(w), "date": week1 + timedelta(days=7 * (int(w) - 1))})
    return pd.DataFrame(rows)


def wiki_daily(title, start, end):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"wiki_{re.sub(r'[^A-Za-z0-9]+', '_', title)}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if f.exists() and end < date.today() - timedelta(days=2):
        return json.loads(f.read_text())
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
           f"all-access/user/{quote(title, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
    for attempt in range(6):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(2 ** attempt)  # rate limited: back off
    if r.status_code == 404:  # article has no views in range
        views = {}
    elif not r.ok:
        print(f"  pageviews {title}: HTTP {r.status_code}", file=sys.stderr)
        return {}  # don't cache failures
    else:
        views = {i["timestamp"][:8]: i["views"] for i in r.json().get("items", [])}
    f.write_text(json.dumps(views))
    time.sleep(0.2)
    return views


def wiki_views(couples, perfs, dates):
    yesterday = date.today() - timedelta(days=1)
    rows = []
    in_week = perfs[["season", "week", "celebrity"]].drop_duplicates()
    for season, cast in couples[couples.season >= FIRST_PAGEVIEW_SEASON].groupby("season"):
        sd = dates[dates.season == season]
        if sd.empty:
            continue
        start, end = sd.date.min() - timedelta(days=14), min(sd.date.max(), yesterday)
        print(f"wiki pageviews season {season}", file=sys.stderr)
        for _, c in cast.iterrows():
            daily = wiki_daily(c["wiki"], start, end) if isinstance(c["wiki"], str) else {}
            for _, wk in sd.iterrows():
                if not ((in_week.season == season) & (in_week.week == wk.week)
                        & (in_week.celebrity == c["celebrity"])).any():
                    continue
                win = [(wk.date - timedelta(days=k)).strftime("%Y%m%d") for k in range(1, 8)]
                vals = [daily[d] for d in win if d in daily]
                rows.append({"season": season, "week": wk.week, "celebrity": c["celebrity"],
                             "wiki_views": sum(vals) / len(vals) * 7 if vals else None})
    return pd.DataFrame(rows)


def parse_views(text):
    m = re.search(r"([\d,]+) views", text or "")
    return int(m.group(1).replace(",", "")) if m else 0


def youtube_views(celebrity, pro, season_year=None, fresh_only=True):
    """Views of official DWTS-channel videos for this couple.

    Returns (total views, median views per video, number of videos, [video dicts]).
    """
    q = f"{celebrity} {pro} Dancing with the Stars {season_year or ''}".strip()
    r = requests.get("https://www.youtube.com/results?search_query=" + quote_plus(q),
                     headers=BROWSER, timeout=30)
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", r.text)
    if not m:
        return None, None, 0, []
    names = [t.lower() for t in celebrity.replace('"', "").split()]
    first, rest = names[0], [t for t in names[1:] if len(t) > 2]
    found = {}

    def walk(o):
        if isinstance(o, dict):
            v = o.get("videoRenderer")
            if v:
                title = "".join(x.get("text", "") for x in v.get("title", {}).get("runs", [])).lower()
                owner = "".join(x.get("text", "") for x in v.get("ownerText", {}).get("runs", []))
                age = v.get("publishedTimeText", {}).get("simpleText", "")
                fresh = not re.search(r"year", age) or not fresh_only
                if (owner.lower().startswith("dancing with the stars") and fresh
                        and first in title and (not rest or any(t in title for t in rest))):
                    found[v["videoId"]] = {
                        "video_id": v["videoId"], "title": title, "published": age,
                        "views": parse_views(v.get("viewCountText", {}).get("simpleText", "")),
                    }
            for x in o.values():
                walk(x)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(json.loads(m.group(1)))
    vals = sorted(f["views"] for f in found.values())
    if not vals:
        return None, None, 0, []
    return sum(vals), vals[len(vals) // 2], len(vals), list(found.values())


def youtube_history(couples, seasons):
    """Median official-video views for past casts (used to learn how much YouTube buzz matters).

    Old videos keep collecting views after the season, so this leaks a little hindsight
    (winners get rewatched). Median per video, compared within a cast, keeps that small.
    """
    f = DATA / "youtube_hist.csv"
    have = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["season", "celebrity"])
    rows = have.to_dict("records")
    done = set(zip(have.season, have.celebrity))
    for _, c in couples[couples.season.isin(seasons)].iterrows():
        if (c.season, c.celebrity) in done:
            continue
        try:
            total, med, n, _ = youtube_views(c["celebrity"], c["pro"].split("  ")[0], fresh_only=False)
        except Exception as e:
            print(f"  youtube failed for {c['celebrity']}: {e}", file=sys.stderr)
            continue
        print(f"  youtube s{c.season} {c['celebrity']}: median {med} over {n}", file=sys.stderr)
        rows.append({"season": c.season, "celebrity": c.celebrity, "yt_median": med, "yt_videos": n})
        pd.DataFrame(rows).to_csv(f, index=False)
        time.sleep(1)
    return pd.DataFrame(rows)


def social_live(couples, season):
    manual_f = DATA / "social_manual.csv"
    cast = couples[couples.season == season]
    if not manual_f.exists():
        pd.DataFrame({"celebrity": cast.celebrity, "instagram": "", "tiktok": ""}).to_csv(manual_f, index=False)
    manual = pd.read_csv(manual_f)
    year = date.today().year
    rows, videos = [], []
    for _, c in cast.iterrows():
        try:
            yt, med, n, vids = youtube_views(c["celebrity"], c["pro"], year)
        except Exception as e:  # network hiccup: keep going, YouTube is optional
            print(f"  youtube failed for {c['celebrity']}: {e}", file=sys.stderr)
            yt, med, n, vids = None, None, 0, []
        videos += [{"celebrity": c["celebrity"], **v} for v in vids]
        print(f"  youtube {c['celebrity']}: {yt} views over {n} videos", file=sys.stderr)
        rows.append({"celebrity": c["celebrity"], "youtube_views": yt, "yt_median": med, "youtube_videos": n})
        time.sleep(1)
    return pd.DataFrame(rows).merge(manual, on="celebrity", how="left"), pd.DataFrame(videos)


def snapshot(season, week, social, videos, wiki):
    """Append today's numbers to the logs instead of overwriting them.

    Over a season this builds point-in-time data: what each couple's views were *before*
    each show, and how fast each new video gains views (which today's totals can't show).
    """
    SNAP.mkdir(parents=True, exist_ok=True)
    taken = datetime.now().isoformat(timespec="minutes")
    w = wiki[(wiki.season == season) & (wiki.week == week)][["celebrity", "wiki_views"]]
    log = social.merge(w, on="celebrity", how="left").assign(taken_at=taken, season=season, week=week)
    cols = ["taken_at", "season", "week", "celebrity", "youtube_views", "yt_median",
            "youtube_videos", "wiki_views", "instagram", "tiktok"]
    append(SNAP / "views_log.csv", log.reindex(columns=cols))
    if len(videos):
        append(SNAP / "youtube_videos.csv", videos.assign(taken_at=taken, season=season)
               .reindex(columns=["taken_at", "season", "celebrity", "video_id", "title", "published", "views"]))


def append(path, df):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    couples = pd.read_csv(DATA / "couples.csv")
    perfs = pd.read_csv(DATA / "performances.csv")
    dates = week_dates(couples, perfs)
    dates.to_csv(DATA / "week_dates.csv", index=False)
    wiki_views(couples, perfs, dates).to_csv(DATA / "wiki_views.csv", index=False)
    if "--no-youtube" not in sys.argv:
        live = couples.season.max()
        youtube_history(couples, range(YT_HISTORY_FROM, live))
        social, videos = social_live(couples, live)
        social.to_csv(DATA / "social_live.csv", index=False)
        week = int(perfs[perfs.season == live].week.max())
        snapshot(live, week, social, videos, pd.read_csv(DATA / "wiki_views.csv"))


if __name__ == "__main__":
    main()
