#!/usr/bin/env python3
"""
sim_rates.py  -  Estimate per-90 player rates from db.sqli for the simulation.

For each player we compute form-weighted, competition/opponent-weighted, and
Bayesian-shrunk per-90 rates for: goals, assists, saves, defensive actions,
yellow/red cards, plus a start-probability and an average minutes-when-playing.

Reads the key-value player_match_stats table produced by scrape_matches.py:
  player_match_stats(fotmob_id, match_id, ..., minutes, section, stat_key,
                     stat_name, value, total, pct, value_type)
  matches(match_id, date, league_name, ...)

Stats we look for by stat_key (FotMob keys):
  goals, assists, saves, minutes_played, yellow_card?, red_card?,
  and defensive actions: tackles, interceptions, clearances, blocks, recoveries.
"""
import math
import sqlite3
from collections import defaultdict

from sim_config import (competition_weight, FORM_DECAY, SHRINKAGE_K,
                        POSITION_PRIORS, OPPONENT_STRENGTH_RANGE)

# stat_keys that count as "defensive actions" (summed per match)
DEF_ACTION_KEYS = {"tackles", "interceptions", "clearances", "blocks",
                   "recoveries", "tackles_won", "ball_recovery"}
# possible card keys (FotMob doesn't always expose; we try several)
YELLOW_KEYS = {"yellow_card", "yellow_cards", "yellow"}
RED_KEYS = {"red_card", "red_cards", "red"}

def _safe(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

def load_player_matches(con, fotmob_id):
    """Return a list of per-match dicts (most-recent first) for one player,
    each with: minutes, league_name, date, and a stat dict {key: value}."""
    cur = con.cursor()
    rows = cur.execute("""
        SELECT pms.match_id, pms.minutes, pms.stat_key, pms.value,
               m.league_name, m.date
        FROM player_match_stats pms
        LEFT JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.fotmob_id = ?
    """, (str(fotmob_id),)).fetchall()
    matches = defaultdict(lambda: {"minutes": 0, "league": None, "date": None, "stats": {}})
    for mid, minutes, key, value, league, date in rows:
        mm = matches[mid]
        if minutes is not None:
            mm["minutes"] = minutes
        mm["league"] = league
        mm["date"] = date
        if key:
            mm["stats"][key] = _safe(value)
    # sort by date desc (most recent first); None dates go last
    out = sorted(matches.values(),
                 key=lambda d: (d["date"] or ""), reverse=True)
    return out

def _match_weight(match, idx, opp_strength=1.0):
    """Combine recency (form decay), competition tier, and opponent strength."""
    recency = FORM_DECAY ** idx
    comp = competition_weight(match.get("league"))
    return recency * comp * opp_strength

def _def_actions(stats):
    return sum(v for k, v in stats.items() if k in DEF_ACTION_KEYS)

def estimate_rates(con, fotmob_id, position):
    """Return a dict of shrunk per-90 rates + start_prob + avg_minutes for a player.
    Uses only matches where the player actually appeared (minutes > 0)."""
    matches = load_player_matches(con, fotmob_id)
    prior = POSITION_PRIORS.get(position, POSITION_PRIORS["MID"])

    # accumulate weighted per-90 contributions
    acc = defaultdict(float)     # stat -> weighted (event/90) sum
    wsum = 0.0                   # total weight over appeared matches
    appeared_w = 0.0             # weighted appearances
    total_w = 0.0                # weight over ALL matches (for start prob)
    minutes_acc = 0.0
    minutes_w = 0.0

    for idx, mm in enumerate(matches):
        w = _match_weight(mm, idx)
        total_w += w
        mins = mm["minutes"] or 0
        if mins <= 0:
            continue  # didn't play -> contributes to start-prob denominator only
        appeared_w += w
        scale = 90.0 / max(mins, 1)
        st = mm["stats"]
        acc["goals"]   += w * _safe(st.get("goals")) * scale
        acc["assists"] += w * _safe(st.get("assists")) * scale
        acc["saves"]   += w * _safe(st.get("saves")) * scale
        acc["def_actions"] += w * _def_actions(st) * scale
        acc["yellow"]  += w * sum(_safe(st.get(k)) for k in YELLOW_KEYS) * scale
        acc["red"]     += w * sum(_safe(st.get(k)) for k in RED_KEYS) * scale
        wsum += w
        minutes_acc += w * mins
        minutes_w += w

    # raw per-90 rates (weighted mean over appeared matches)
    raw = {k: (acc[k] / wsum if wsum else prior.get(k, 0.0))
           for k in ("goals", "assists", "saves", "def_actions", "yellow", "red")}

    # Bayesian shrinkage toward position prior:
    # shrunk = (n*raw + K*prior)/(n+K), n = effective appeared matches
    n_eff = appeared_w
    def shrink(stat, k_key):
        K = SHRINKAGE_K.get(k_key, 6)
        p = prior.get(stat, 0.0)
        return (n_eff * raw[stat] + K * p) / (n_eff + K) if (n_eff + K) else p
    rates = {
        "goals_p90":   shrink("goals", "goals"),
        "assists_p90": shrink("assists", "assists"),
        "saves_p90":   shrink("saves", "saves"),
        "def_actions_p90": shrink("def_actions", "def_actions"),
        "yellow_p90":  shrink("yellow", "cards"),
        "red_p90":     shrink("red", "cards"),
    }

    # start probability: weighted fraction of matches with 60+ minutes,
    # shrunk lightly toward 0.7 prior so zero-sample players aren't 0 or 1.
    start_events = 0.0
    for idx, mm in enumerate(matches):
        w = _match_weight(mm, idx)
        if (mm["minutes"] or 0) >= 60:
            start_events += w
    K_start = 4
    start_prob = (start_events + K_start * 0.65) / (total_w + K_start) if total_w else 0.65
    avg_minutes = (minutes_acc / minutes_w) if minutes_w else 70.0

    return {
        "fotmob_id": str(fotmob_id),
        "position": position,
        "n_matches": len(matches),
        "n_appeared": int(round(appeared_w)),
        "start_prob": round(min(max(start_prob, 0.02), 0.99), 3),
        "avg_minutes": round(min(max(avg_minutes, 1), 95), 1),
        **{k: round(v, 4) for k, v in rates.items()},
    }

def estimate_team_goal_shares(con, players_in_team):
    """Given a list of (fotmob_id, position) for one nation, return a normalized
    goal-share dict so the team's expected goals can be allocated to players.
    Goal share ∝ goals_p90 * start_prob * avg_minutes/90 (expected goals/ match)."""
    weights = {}
    for fid, pos in players_in_team:
        r = estimate_rates(con, fid, pos)
        exp_goals = r["goals_p90"] * r["start_prob"] * (r["avg_minutes"] / 90.0)
        weights[str(fid)] = max(exp_goals, 1e-6)
    tot = sum(weights.values())
    return {k: v / tot for k, v in weights.items()} if tot else {}

if __name__ == "__main__":
    import sys, json
    db = sys.argv[1] if len(sys.argv) > 1 else "db.sqli"
    fid = sys.argv[2] if len(sys.argv) > 2 else None
    con = sqlite3.connect(db)
    if fid:
        pos = con.execute("SELECT position FROM players WHERE fotmob_id=?",
                          (fid,)).fetchone()
        pos = pos[0] if pos else "MID"
        print(json.dumps(estimate_rates(con, fid, pos), indent=2))
    else:
        n = con.execute("SELECT COUNT(*) FROM player_match_stats").fetchone()[0]
        print(f"player_match_stats rows: {n}")
    con.close()
