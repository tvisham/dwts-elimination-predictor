"""Scrape every DWTS (US) season from Wikipedia into tidy CSVs.

Outputs
  data/couples.csv      season, couple, celebrity, pro, notability, status, place
  data/performances.csv season, week, theme, couple, score, n_judges, dance, result
"""
import re
import sys
import time
from urllib.parse import unquote
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
URL = "https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)_season_{n}"
HEADERS = {"User-Agent": "dwts-predictor/0.1 (personal hobby project)"}
CURRENT_SEASON = 35


def fetch(n, refresh=False):
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"s{n}.html"
    if refresh or not f.exists() or f.stat().st_size == 0:
        r = requests.get(URL.format(n=n), headers=HEADERS, timeout=30)
        r.raise_for_status()
        f.write_text(r.text, encoding="utf8")
        time.sleep(0.5)
    return BeautifulSoup(f.read_text(encoding="utf8"), "lxml")


def clean(text):
    text = re.sub(r"\[\s*[^\]]*\]", "", text)  # footnote markers [ a ], [ 12 ]
    return re.sub(r"\s+", " ", text).replace("†", "").replace("‡", "").strip()


def grid(table):
    """Expand a wikitable into a rectangular list of rows, resolving row/colspans."""
    out, pending = [], {}  # pending[(row, col)] = text
    for ri, tr in enumerate(table.find_all("tr")):
        row, ci = [], 0
        cells = tr.find_all(["th", "td"], recursive=False)
        for cell in cells:
            while (ri, ci) in pending:
                row.append(pending.pop((ri, ci)))
                ci += 1
            text = clean(cell.get_text(" ", strip=True))
            rs = int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1)
            for c in range(cs):
                row.append(text)
                for r in range(1, rs):
                    pending[(ri + r, ci + c)] = text
            ci += cs
        while (ri, ci) in pending:
            row.append(pending.pop((ri, ci)))
            ci += 1
        out.append(row)
    return out


def first_num(text):
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def parse_cast(soup, season):
    table = soup.select("table.wikitable")[0]
    rows = grid(table)
    head = [h.lower() for h in rows[0]]
    ci = next(i for i, h in enumerate(head) if "celebrity" in h)
    pi = next(i for i, h in enumerate(head) if "partner" in h)
    si = next(i for i, h in enumerate(head) if "status" in h)
    ni = next((i for i, h in enumerate(head) if "notab" in h), None)
    links = {}
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["th", "td"], recursive=False)
        a = cells[0].find("a", href=True) if cells else None
        if a and "/wiki/" in a["href"]:
            links[clean(cells[0].get_text(" ", strip=True))] = unquote(a["href"].split("/wiki/")[1])
    by_text = {}  # any linked name in the table -> its article (used for pro partners)
    for a in table.find_all("a", href=True):
        if "/wiki/" in a["href"]:
            by_text.setdefault(clean(a.get_text(" ", strip=True)), unquote(a["href"].split("/wiki/")[1]))
    out = []
    for r in rows[1:]:
        if len(r) <= si:
            continue
        pro_link = next((v for k, v in by_text.items() if k and r[pi].startswith(k)), "")
        celeb = re.sub(r"\s*Season \d+.*$", "", r[ci]).strip()  # all-stars "Name Season 4"
        out.append({
            "season": season,
            "celebrity": celeb,
            "pro": r[pi],
            "notability": r[ni] if ni is not None else "",
            "status": r[si],
            "wiki": links.get(r[ci], ""),
            "pro_wiki": pro_link,
        })
    return out


def season_weeks(soup):
    """Number of weeks in the season, read from the scoring chart's header (1, 2, ... N)."""
    for t in soup.select("table.wikitable"):
        cap = t.caption.get_text(" ", strip=True) if t.caption else ""
        if "Weekly scores" in cap:
            head = []
            for r in grid(t):
                if any("&" in c for c in r):  # first couple row: header is over
                    break
                head += r
            nums = [int(c) for c in head if re.fullmatch(r"\d{1,2}", c)]
            return max(nums) if nums else None
    return None


def week_tables(soup):
    """Yield (week_no, theme, table) for every table under the Weekly scores section."""
    start = soup.find(id=re.compile(r"^Weekly_scores")) or soup.find(id=re.compile(r"^Weekly"))
    if start is None:
        return
    start = start.find_parent(["h2", "div"]) or start
    for table in start.find_all_next("table", class_="wikitable"):
        h = table.find_previous(["h3", "h4"])
        h2 = table.find_previous("h2")
        if h2 and re.search(r"Dance chart|References|Notes|Ratings", h2.get_text()):
            break
        if not h:
            continue
        title = clean(h.get_text(" ", strip=True))
        m = re.match(r"Week (\d+)\s*(?:\(.*?\))?\s*:?\s*(.*)", title)
        if m:
            yield int(m.group(1)), m.group(2).strip() or f"Week {m.group(1)}", table


def parse_weeks(soup, season, couple_names):
    out = []
    for week, theme, table in week_tables(soup):
        rows = grid(table)
        if not rows:
            continue
        # header may span two rows (judge-name sub-header)
        head = [h.lower() for h in rows[0]]
        body_start = 2 if len(rows) > 1 and not any(_is_couple(x, couple_names) for x in rows[1][:2]) else 1
        score_i = next((i for i, h in enumerate(head) if "total" in h), None)
        if score_i is None:
            score_i = next((i for i, h in enumerate(head) if "score" in h), None)
        res_i = next((i for i, h in enumerate(head) if "result" in h), None)
        dance_i = next((i for i, h in enumerate(head) if h.startswith("dance")), None)
        couple_i = next((i for i, h in enumerate(head) if "couple" in h), 0)
        judge_cols = [i for i, h in enumerate(head) if h.startswith("score") and score_i != i]
        if score_i is None:
            continue
        for r in rows[body_start:]:
            if len(r) <= score_i:
                continue
            couple = r[couple_i]
            if not _is_couple(couple, couple_names):
                continue
            score = first_num(r[score_i])  # None = hasn't danced yet (live week)
            judges = re.search(r"\(([\d.,\s]+)\)", r[score_i])
            n_j = len(judges.group(1).split(",")) if judges else max(len(judge_cols), 3)
            out.append({
                "season": season, "week": week, "theme": theme, "couple": couple,
                "score": score, "n_judges": n_j,
                "dance": r[dance_i] if dance_i is not None and dance_i < len(r) else "",
                "result": r[res_i] if res_i is not None and res_i < len(r) else "",
            })
    return out


def norm(token):
    return re.sub(r"[^a-z]", "", token.lower())


def name_tokens(celebrity):
    toks = {norm(t) for t in re.split(r"[\s.]+", celebrity.replace('"', " ")) if norm(t)}
    initials = norm("".join(re.findall(r"\b[A-Z]\.", celebrity)))  # "J. R." -> "jr"
    return toks | ({initials} if len(initials) > 1 else set())


def _is_couple(text, names):
    if (text or "").count("&") != 1:
        return False
    return norm(text.split("&")[0].strip().split(" ")[0]) in names


def resolve(perfs, cast):
    """Attach each 'Amber & Pasha' performance row to its cast member.

    Labels change mid-season ('Connor W. & Rylee' -> 'Connor & Rylee'), so match on the
    celebrity's name tokens plus the pro's first name rather than on the exact string.
    """
    out = []
    for p in perfs:
        celeb_part, pro_part = [x.strip() for x in p["couple"].split("&")]
        first = norm(celeb_part.split(" ")[0])
        pro_first = norm(pro_part.split(" ")[0])
        cands = [c for c in cast if first in name_tokens(c["celebrity"])]
        if len(cands) > 1:
            cands = [c for c in cands if pro_first in {norm(t) for t in c["pro"].split()}] or cands
        if len(cands) > 1 and celeb_part.endswith("."):
            cands = [c for c in cands if c["celebrity"].split(" ")[-1][0] == celeb_part[-2]] or cands
        if cands:
            p["celebrity"] = cands[0]["celebrity"]
            out.append(p)
    return out


def place_of(status, n_cast):
    s = status.lower()
    if "winner" in s:
        return 1
    if "runner" in s:
        return 2
    if "third" in s:
        return 3
    if "fourth" in s:
        return 4
    if "fifth" in s:
        return 5
    m = re.search(r"eliminated (\d+)", s)
    if m:
        return n_cast - int(m.group(1)) + 1
    return None


def main(refresh_current=True):
    all_cast, all_perf, meta = [], [], []
    for n in range(1, CURRENT_SEASON + 1):
        soup = fetch(n, refresh=(refresh_current and n == CURRENT_SEASON))
        cast = parse_cast(soup, n)
        firsts = set().union(*(name_tokens(c["celebrity"]) for c in cast))
        perfs = parse_weeks(soup, n, firsts)
        perfs = resolve(perfs, cast)
        label = {}
        for p in perfs:  # most recent label wins, e.g. 'Connor & Rylee'
            label[p["celebrity"]] = p["couple"]
        for c in cast:
            c["couple"] = label.get(c["celebrity"])
        for c in cast:
            c["place"] = place_of(c["status"], len(cast))
        # judges'-save era: weekly tables carry a "Judges' votes to save" column
        save = len(re.findall(r"Judges.{0,3} votes? to save", soup.get_text(" ")))
        meta.append({"season": n, "weeks": season_weeks(soup), "judges_save": int(save >= 3)})
        all_cast += cast
        all_perf += perfs
        print(f"season {n:>2}: {len(cast):>2} couples, {len(perfs):>3} performances", file=sys.stderr)
    pd.DataFrame(all_cast).to_csv(ROOT / "data" / "couples.csv", index=False)
    pd.DataFrame(all_perf).to_csv(ROOT / "data" / "performances.csv", index=False)
    pd.DataFrame(meta).to_csv(ROOT / "data" / "seasons.csv", index=False)


if __name__ == "__main__":
    main(refresh_current="--no-refresh" not in sys.argv)
