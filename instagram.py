"""Instagram likes on @dancingwiththestars performance posts, the model's viewer-vote proxy.

Two ways in:

1. Official Meta Graph API (Business Discovery), used when these environment variables
   are set:
       IG_USER_ID       your own Instagram Business/Creator account id
       IG_ACCESS_TOKEN  a Meta app access token for that account
   It pages back through @dancingwiththestars' posts, matches each caption to a couple,
   and assigns it to a show week by its timestamp. This reaches every week of the season.

2. data/instagram_posts.csv, filled by hand (or by anyone reading the number under a
   post): season, week, celebrity, url, likes, checked_at, note. Without a token this is
   the only source; Instagram shows logged-out visitors only the newest posts.

Likes keep growing for days, so the model only compares couples *within* a week (as
ranks), where every post had about the same time to collect likes.

Outputs
  data/instagram_posts.csv           every post with its latest like count
  data/instagram_likes.csv           season, week, celebrity, posts, likes (mean per post)
  data/snapshots/instagram_log.csv   every reading, timestamped
"""
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
POSTS = DATA / "instagram_posts.csv"
TARGET = "dancingwiththestars"
GRAPH = "https://graph.facebook.com/v21.0"
COLS = ["season", "week", "celebrity", "url", "likes", "checked_at", "note"]


def load_posts():
    return pd.read_csv(POSTS) if POSTS.exists() else pd.DataFrame(columns=COLS)


def graph_media(user_id, token, since):
    """All of the target's posts newer than `since`, via Business Discovery."""
    after, out = None, []
    while True:
        page = f".after({after})" if after else ""
        fields = (f"business_discovery.username({TARGET})"
                  f"{{media{page}.limit(50){{caption,like_count,comments_count,timestamp,permalink}}}}")
        r = requests.get(f"{GRAPH}/{user_id}", params={"fields": fields, "access_token": token}, timeout=30)
        r.raise_for_status()
        media = r.json()["business_discovery"]["media"]
        out += media["data"]
        oldest = min(datetime.fromisoformat(m["timestamp"][:19]) for m in media["data"])
        after = media.get("paging", {}).get("cursors", {}).get("after")
        if not after or oldest.date() < since:
            return out


def match_couple(caption, cast):
    """The one couple a caption is about (posts naming several couples are skipped)."""
    text = (caption or "").lower()
    hits = []
    for _, c in cast.iterrows():
        first = c["celebrity"].split()[0].lower()
        pro_first = str(c["pro"]).split()[0].lower()
        if re.search(rf"\b{re.escape(first)}\b", text) and re.search(rf"\b{re.escape(pro_first)}\b", text):
            hits.append(c["celebrity"])
    return hits[0] if len(hits) == 1 else None


def week_of(day, dates):
    """Show week a post belongs to: the latest air date on or before the post (UTC slack of a day)."""
    aired = dates[dates.date <= day + timedelta(days=1)]
    return int(aired.week.max()) if len(aired) else None


def from_api(couples, dates, season):
    user_id, token = os.environ.get("IG_USER_ID"), os.environ.get("IG_ACCESS_TOKEN")
    if not (user_id and token):
        return None
    cast = couples[couples.season == season]
    sd = dates[dates.season == season].copy()
    sd["date"] = pd.to_datetime(sd.date).dt.date
    rows = []
    for m in graph_media(user_id, token, sd.date.min() - timedelta(days=2)):
        if "like_count" not in m:  # owner hid the count
            continue
        celeb = match_couple(m.get("caption"), cast)
        day = datetime.fromisoformat(m["timestamp"][:19]).date()
        week = week_of(day, sd)
        if celeb and week:
            rows.append({"season": season, "week": week, "celebrity": celeb, "url": m["permalink"],
                         "likes": m["like_count"], "checked_at": datetime.now().isoformat(timespec="minutes"),
                         "note": (m.get("caption") or "")[:60].replace("\n", " ")})
    print(f"  instagram api: {len(rows)} couple posts matched", file=sys.stderr)
    return pd.DataFrame(rows, columns=COLS)


def main():
    couples = pd.read_csv(DATA / "couples.csv")
    dates = pd.read_csv(DATA / "week_dates.csv")
    season = int(couples.season.max())
    posts = load_posts()
    try:
        fresh = from_api(couples, dates, season)
    except requests.RequestException as e:
        print(f"  instagram api failed ({e}); using {POSTS.name} only", file=sys.stderr)
        fresh = None
    if fresh is not None and len(fresh):
        # newest reading of each post wins; hand-entered posts the API didn't see are kept
        posts = pd.concat([posts, fresh]).drop_duplicates("url", keep="last")
        snap = DATA / "snapshots" / "instagram_log.csv"
        snap.parent.mkdir(exist_ok=True)
        fresh.to_csv(snap, mode="a", header=not snap.exists(), index=False)
    elif fresh is None:
        print("  instagram: no IG_USER_ID/IG_ACCESS_TOKEN set, using hand-entered posts", file=sys.stderr)
    posts.to_csv(POSTS, index=False)

    likes = (posts.groupby(["season", "week", "celebrity"])
             .agg(posts=("likes", "size"), likes=("likes", "mean")).reset_index())
    likes.to_csv(DATA / "instagram_likes.csv", index=False)
    print(f"  instagram: {len(likes)} couple-weeks over weeks {sorted(likes.week.unique().tolist())}", file=sys.stderr)


if __name__ == "__main__":
    main()
