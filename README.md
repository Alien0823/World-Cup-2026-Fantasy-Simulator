# World Cup 2026 Fantasy Simulator

A Monte-Carlo simulator and squad optimiser for **FIFA World Cup Fantasy** and
**SofaScore Fantasy**. It combines bookmaker odds (de-vigged) with FotMob player
form to project fantasy points, rate your squad, and solve for the optimal team.

![status](https://img.shields.io/badge/build-experimental-yellow)

## What it does

- Simulates each fixture with a **Dixon-Coles bivariate Poisson** model driven by
  de-vigged bookmaker odds (the odds set each team's expected goals).
- Allocates goals to players via **form-weighted, normalised goal-share** derived
  from per-90 rates in their last ~1.5 seasons of detailed match stats.
- Scores every simulation under the **official FIFA and SofaScore rules**
  (goals, assists, clean sheets, cards, saves, defensive actions, bonuses,
  SofaScore rating bands).
- Reports per-player expected points, P(goal), P(clean sheet), and a whole-squad
  points **distribution + team rating (0–100 and a letter grade)**.
- Finds the **optimal legal squad** with exact integer programming (budget,
  formation, max-3-per-nation, captaincy).

## Quick start (just the app)

Download Database from:https://limewire.com/d/XXt7u#F2lqVCQknf and put it inside folder

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Building the data from scratch

The app reads two files: `db.sqli` (players + match stats) and `matches.json`
(fixture odds). Build them with the pipeline below. Some steps scrape FotMob,
which needs a one-time browser install:

```bash
pip install -r requirements.txt
playwright install chromium
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit GUI |
| `sim_engine.py` | Monte-Carlo simulation (Dixon-Coles, scoring) |
| `sim_rates.py` | Per-90 rate estimation from `db.sqli` |
| `sim_config.py` | Scoring systems + game rules (FIFA & SofaScore) |
| `sim_optimize.py` | Exact ILP squad optimiser |
| `parse_odds.py` | De-vig bookmaker odds → `matches.json` |
| `parse_squads.py` | ESPN squad HTML → roster |
| `fotmob_resolver.py` | Resolve FotMob player IDs |
| `fotmob_scraper.py` | Season stats per player |
| `scrape_matches.py` | Detailed per-match stats → SQLite |
| `scrape_fifa_prices.py` | FIFA Fantasy prices (live API) |
| `parse_sofascore_prices.py` | SofaScore Fantasy prices (saved HTML) |

## Notes & caveats
- If price or position of player is wrong use https://sqlable.com/sqlite/# to update data in DB. Save it and run app again
- Projections are only as good as the inputs (odds + form). Treat them as a
  decision aid, not a guarantee.
- FotMob and SofaScore have no official public API; the scrapers use internal
  endpoints / saved pages and may need adjustment if those sites change.
- This is a hobby project for fantasy planning, not betting advice.

## Licence

MIT — do what you like, no warranty.
