# DWTS Elimination Predictor

Predicts who goes home each week of *Dancing with the Stars*. It combines judges' scores, form, pro partner, and how many views each celebrity gets, then simulates the rest of the season thousands of times (the same approach as the World Cup ML predictors).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python run.py
python -m http.server 8765 --directory web
```

Then open http://localhost:8765.

On macOS/Linux, use `.venv/bin/` instead of `.venv/Scripts/`.

Re-run `run.py` after each show. It pulls the new scores and results from Wikipedia, refreshes the view counts, retrains the model, and rewrites `web/data.js`.

## How it works

| Step | File | What it does |
|------|------|--------------|
| 1 | `scrape.py` | Reads every season's Wikipedia page and writes `data/couples.csv` (cast, pro, finish) and `data/performances.csv` (week, theme night, score, dance, result) |
| 2 | `popularity.py` | Collects Wikipedia pageviews from the 7 days before each show (season 21+) and median views per official DWTS YouTube performance video (season 34 + the live season) |
| 3 | `model.py` | Fits the model on every past elimination, backtests it, runs a 20,000-season simulation, and writes `web/data.js` |
| UI | `web/index.html` | Danger zone, projected elimination path, week-by-week odds table, couple cards, and track record |

## What the model looks at

- **Judges' scores**: this week vs. the rest of the field, plus prior form and trend
- **Bottom-two history**: how many times a couple has already been in the bottom two
- **Escapes**: weeks a couple survived despite a bottom-two judges' score (a sign of strong fan votes)
- **Pro partner**: how that pro's past celebrities finished
- **YouTube views**: views per official performance video (the strongest popularity signal)
- **Wikipedia pageviews**: included, but only a weak signal
- **Instagram / TikTok**: these sites block scraping, so add follower counts by hand in `data/social_manual.csv` and they get blended into the YouTube signal

The model is a conditional logit: each elimination night is treated as a race to the bottom, and every couple's odds for that week add up to 100%.

## Accuracy

Backtested on seasons 21–34 (102 elimination weeks). Each season was predicted using only the seasons before it.

| Method | Picked who went home |
|--------|----------------------|
| **This model** | **48%** (eliminated couple was in its bottom 3: 83%) |
| Lowest judges' score goes home | 41% |
| Random guess | 14% |

## Caveats

- YouTube history only goes back to season 34, and older videos keep collecting views, so the YouTube weight is learned from a small sample.
- The finale winner is ranked by buzz plus form, because there's too little finale data to learn from.
- Season length (`SEASON_WEEKS`) and number of finalists (`FINALISTS`) are set at the top of `model.py`.

## Data sources

- Wikipedia season pages for Dancing with the Stars (US), seasons 1–35
- Wikimedia Pageviews API
- YouTube search results (official Dancing With The Stars channel)
