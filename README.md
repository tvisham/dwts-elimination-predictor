# DWTS Elimination Predictor 🪩

Predicts who goes home each week of *Dancing with the Stars*. It combines judges' scores, form,
pro partner and how many views each celebrity gets, then simulates the rest of the season
(the same approach as the World Cup ML predictors).

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # macOS/Linux: .venv/bin/pip
.venv/Scripts/python run.py                        # scrape → views → train → simulate
python -m http.server 8765 --directory web         # then open http://localhost:8765
```

Re-run `run.py` after each show. It picks up the new scores and results from Wikipedia,
refreshes the view counts, retrains, and rewrites `web/data.js`.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `scrape.py` | Every season's Wikipedia page → `data/couples.csv` (cast, pro, finish) and `data/performances.csv` (week, theme night, score, dance, result) |
| 2 | `popularity.py` | Wikipedia pageviews in the 7 days before each show (seasons 21+), and median views per official DWTS YouTube performance video (season 34 + live season) |
| 3 | `model.py` | Conditional-logit model fit on every past elimination, backtest, 20,000-season Monte Carlo → `web/data.js` |
| UI | `web/index.html` | Danger zone, projected elimination path, week-by-week odds table, couple cards, track record |

### Features (fit on seasons 1–34 + this season's finished weeks)
- Judges' score this week vs. the field, prior form, trend
- Times in the bottom two; **escapes**: weeks survived with a bottom-two judges' score (fan votes)
- Pro partner's historical finishes
- **Views**: YouTube views per official performance video (strongest popularity signal), Wikipedia pageviews (weak)
- Instagram/TikTok: those sites block scraping, so add follower counts by hand in
  `data/social_manual.csv` and they get blended into the YouTube signal

## Accuracy
Backtest over seasons 21–34 (102 elimination weeks). Each season was predicted using only earlier seasons:

| | Correct pick |
|---|---|
| This model | **48%** (eliminated couple was in its bottom 3: 83%) |
| "Lowest judges' score goes home" | 41% |
| Random guess | 14% |

Caveats: YouTube history only goes back to season 34, and older videos keep collecting views,
so the YouTube weight comes from a small sample. The finale winner is ranked by buzz plus form,
because there's too little finale data to fit.

Season length (`SEASON_WEEKS`) and finalist count (`FINALISTS`) are set at the top of `model.py`.
