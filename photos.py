"""Download a photo of every celebrity and pro in the live season.

Photos come from each person's Wikipedia article (the lead image, hosted on Wikimedia
Commons, which is freely licensed). Official ABC couple photos are copyrighted, so
the page shows the two Commons portraits side by side instead.

Outputs
  web/img/<slug>.jpg     thumbnails (about 330 px)
  web/img/custom/        couple photos you saved yourself (celebrity and pro together), named
                         after the celebrity, e.g. connor-wood.jpg; shown instead of the portraits
  data/photos.csv        person, role, file, article, image page (for credits)
"""
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMG = ROOT / "web" / "img"
UA = {"User-Agent": "dwts-predictor/0.1 (personal hobby project)"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def summary(title):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title, safe="")
    for attempt in range(5):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 429:
            break
        time.sleep(2 ** attempt)
    return r.json() if r.ok else {}


def is_about(person, page):
    """The cast table sometimes links a show instead of the person (e.g. a reality series)."""
    title = page.get("title", "").lower()
    tokens = [t for t in re.split(r"[\s.\"]+", person.lower()) if len(t) > 2]
    return page.get("type") == "standard" and any(t in title for t in tokens)


def couple_photo(celebrity):
    """A couple photo you saved into web/img/custom/<celebrity>.jpg, if there is one."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        f = IMG / "custom" / f"{slug(celebrity)}.{ext}"
        if f.exists():
            return {"person": celebrity, "role": "couple", "file": f"img/custom/{f.name}",
                    "article": "", "image": ""}
    return None


def fetch(person, article, role):
    if not isinstance(article, str) or not article:
        return None
    page = summary(article)
    thumb = page.get("thumbnail", {}).get("source")
    if not thumb or not is_about(person, page):
        return None
    IMG.mkdir(parents=True, exist_ok=True)
    out = IMG / f"{slug(person)}.jpg"
    if not out.exists():
        r = requests.get(thumb, headers=UA, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        time.sleep(0.3)
    original = page.get("originalimage", {}).get("source", thumb)
    return {"person": person, "role": role, "file": f"img/{out.name}",
            "article": f"https://en.wikipedia.org/wiki/{article}",
            "image": "https://commons.wikimedia.org/wiki/File:" + original.rsplit("/", 1)[-1]}


def main():
    couples = pd.read_csv(DATA / "couples.csv")
    cast = couples[couples.season == couples.season.max()]
    rows = []
    for _, c in cast.iterrows():
        pro = c["pro"].split("  ")[0]
        both = couple_photo(c["celebrity"])
        if both:
            rows.append(both)
        for person, article, role in [(c["celebrity"], c["wiki"], "celebrity"),
                                      (pro, c.get("pro_wiki"), "pro")]:
            try:
                got = fetch(person, article, role)
            except requests.RequestException as e:
                print(f"  photo failed for {person}: {e}", file=sys.stderr)
                got = None
            print(f"  {'ok  ' if got else 'none'} {person}", file=sys.stderr)
            if got:
                rows.append(got)
    pd.DataFrame(rows).to_csv(DATA / "photos.csv", index=False)


if __name__ == "__main__":
    main()
