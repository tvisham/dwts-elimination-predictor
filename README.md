# DWTS Elimination Predictor

Predicts who goes home each week of *Dancing with the Stars*. It combines judges' scores, form, pro partner, inferred fan support and how many views each celebrity gets, then simulates the rest of the season thousands of times (the same approach as the World Cup ML predictors).

## Quick start

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python run.py
python -m http.server 8765 --directory web
```

Then open http://localhost:8765.

On macOS/Linux, use `.venv/bin/` instead of `.venv/Scripts/`.

Run `run.py` before each show (to freeze that week's odds) and again after it (to grade them and pick up the new scores).

## How it works

| Step | File | What it does |
|------|------|--------------|
| 1 | `scrape.py` | Reads every season's Wikipedia page into `data/couples.csv`, `data/performances.csv` and `data/seasons.csv` (season length, judges'-save era) |
| 2 | `popularity.py` | Collects Wikipedia pageviews and official DWTS YouTube views, and appends every reading to `data/snapshots/` |
| 3 | `photos.py` | Downloads each celebrity's and pro's photo from Wikipedia/Wikimedia Commons into `web/img/` |
| 4 | `model.py` | Fits the model, backtests it, simulates the season 20,000 times, freezes this week's odds in `predictions/`, and writes `web/data.js` |
| UI | `web/index.html` | Danger zone, projected elimination path, season odds, couple cards with photos, calibration chart and live record |

## What the model looks at

- **Judges' scores**: this week vs. the field, prior form and trend, with extra weight in judges'-save seasons (28–31)
- **Bottom-two history**: times a couple has already been in the bottom two
- **Fan rating**: an Elo-style rating that rises each week a couple survives a score the judges alone would have sent home
- **Pro partner**: how that pro's past celebrities finished
- **Celebrity type**: athlete, reality star, musician, social media or actor/TV, read from the cast table
- **YouTube views**: views per official performance video
- **Wikipedia pageviews**: included, but only a weak signal
- **Instagram / TikTok**: these sites block scraping, so add follower counts by hand in `data/social_manual.csv`
- **Dance style**: when a week's dances are announced, the projected score shifts by that style's historical effect

The model is a conditional logit: each elimination night is a race to the bottom, and every couple's odds for that week add up to 100%.

## Accuracy

Backtested on seasons 21–34 (102 elimination weeks). Each season was predicted using only the seasons before it, and the one tuning setting (ridge penalty) was chosen on seasons 11–20, so the test seasons were never used to make choices.

| Method | Correct pick | Log loss (lower is better) |
|--------|--------------|----------------------------|
| **This model** | **41%** (in its bottom 3: 80%) | **1.46** |
| Judges' score only | 38% | 1.56 |
| Lowest judges' score goes home | 35% | — |
| Random guess | 14% | 1.94 |

On double-elimination weeks a pick gets partial credit for each eliminated couple it named.

## Live record

Every run before a show saves that week's odds to `predictions/s35_wNN_preshow.json` (and `_scores.json` once judges' scores are in, before the result). After the result, `model.py` grades them and the page shows the record. Only weeks frozen this way are graded, so the live record can't be fudged after the fact.

## Caveats

- YouTube history only goes back to season 34, and older videos keep collecting views. The snapshot log fixes this going forward.
- The finale winner is ranked by buzz plus form, because there's too little finale data to learn from.
- Six people in season 35 have no photo on Wikipedia, so the page shows their initials.

## Data sources

- Wikipedia season pages for Dancing with the Stars (US), seasons 1–35
- Wikimedia Pageviews API and Wikimedia Commons (photos)
- YouTube search results (official Dancing With The Stars channel)
