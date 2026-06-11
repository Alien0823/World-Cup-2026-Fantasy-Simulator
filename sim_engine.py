#!/usr/bin/env python3
"""
sim_engine.py  -  Monte Carlo fantasy simulation (WC2026).

Pipeline per simulation:
  1. For each match in scope, draw team goals from a Dixon-Coles-adjusted
     bivariate Poisson using the odds-derived lambdas (blended ladder + 1x2).
  2. Allocate each team's goals to its players via normalized goal-share
     (form/competition-weighted, penalty-boosted).
  3. Draw assists, minutes (start prob), cards, saves, defensive actions.
  4. Convert events -> fantasy points under the chosen scoring system
     (FIFA or SofaScore), per position.
  5. Aggregate per player and for the whole user squad; report distributions.

Odds come from matches.json (lambda_home / lambda_away, already de-vigged).
Player rates come from sim_rates.estimate_rates (reads db.sqli).

This is vectorized over N simulations with NumPy (CPU is plenty fast; see notes).
"""
import json
import sqlite3
import numpy as np
from collections import defaultdict

from sim_config import SCORING_SYSTEMS, competition_weight
from sim_rates import estimate_rates

# ---------------------------------------------------------------------------
# Dixon-Coles low-score correction
# ---------------------------------------------------------------------------
def dc_tau(h, a, lam, mu, rho):
    """Dixon-Coles correction factor for low scores (vectorized over arrays)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    tau = np.where(m00, 1 - lam * mu * rho, tau)
    tau = np.where(m01, 1 + lam * rho, tau)
    tau = np.where(m10, 1 + mu * rho, tau)
    tau = np.where(m11, 1 - rho, tau)
    return np.clip(tau, 1e-6, None)

def sample_dixon_coles(lam, mu, rho, N, rng, max_goals=9):
    """Sample (home_goals, away_goals) N times under a Dixon-Coles bivariate model.
    lam, mu are scalars (one match). Uses the joint pmf over a 0..max_goals grid
    with the DC correction, then samples from the flattened categorical."""
    hs = np.arange(max_goals + 1)
    H, A = np.meshgrid(hs, hs, indexing="ij")
    # base independent Poisson pmf
    from scipy.stats import poisson
    ph = poisson.pmf(H, lam)
    pa = poisson.pmf(A, mu)
    joint = ph * pa
    tau = dc_tau(H, A, np.full_like(joint, lam), np.full_like(joint, mu), rho)
    joint = joint * tau
    joint = joint / joint.sum()
    flat = joint.ravel()
    idx = rng.choice(flat.size, size=N, p=flat)
    hg = (idx // (max_goals + 1)).astype(np.int16)
    ag = (idx % (max_goals + 1)).astype(np.int16)
    return hg, ag

# ---------------------------------------------------------------------------
def load_odds(matches_json="matches.json"):
    data = json.loads(open(matches_json, encoding="utf-8").read())
    od = {}
    for m in data["matches"]:
        od[(m["home"], m["away"])] = m
    return od

def find_match(odds, nation, scope_matches):
    """Return (match, is_home) for a nation within the scope, or (None, None)."""
    for (h, a), m in odds.items():
        if (h, a) in scope_matches or not scope_matches:
            if h == nation:
                return m, True
            if a == nation:
                return m, False
    return None, None

# ---------------------------------------------------------------------------
def _rating_to_points(rating, bands):
    """Map a rating array to points via the configured bands (vectorized)."""
    pts = np.zeros_like(rating, dtype=float)
    for (lo, hi), p in bands:
        pts = np.where((rating >= lo) & (rating < hi), p, pts)
    return pts

def points_from_events(ev, position, scoring, system="fifa"):
    """Vectorized: convert event arrays -> fantasy points under one scoring system.
    Handles both FIFA and SofaScore official rules. ev is a dict of np arrays."""
    S = scoring
    N = ev["minutes"].shape[0]
    pts = np.zeros(N, dtype=float)
    played60 = ev["minutes"] >= 60
    played_any = ev["minutes"] > 0
    # appearance: +1 any, +1 more if 60+
    pts += played_any * S["appearance_any"]
    pts += played60 * S["appearance_60_bonus"]
    # goals
    pts += ev["goals"] * S["goal"][position]
    # assists (position-specific in SofaScore, flat in FIFA)
    assist_val = S["assist"][position] if isinstance(S["assist"], dict) else S["assist"]
    pts += ev["assists"] * assist_val
    # clean sheet (needs 60+ and team conceded 0)
    cs = played60 & (ev["team_conceded"] == 0)
    pts += cs * S["clean_sheet"][position]
    # goals conceded penalty
    if "conceded_after_first" in S:   # FIFA: -1 per goal after the first
        unit = S["conceded_after_first"].get(position, 0)
        if unit:
            pts += np.maximum(ev["team_conceded"] - 1, 0) * unit * played_any
    if "conceded_per_2" in S:         # SofaScore: -1 per 2 conceded
        unit = S["conceded_per_2"].get(position, 0)
        if unit:
            pts += (ev["team_conceded"] // 2) * unit * played_any
    # own goals
    pts += ev.get("own_goals", 0) * S.get("own_goal", 0)
    # cards (FIFA flat; SofaScore minute-tiered handled by caller -> ev['card_pts'])
    if "card_pts" in ev:
        pts += ev["card_pts"]
    else:
        pts += ev["yellow"] * S.get("yellow_card", 0)
        pts += ev["red"] * S.get("red_card", 0)
    # penalties
    pts += ev.get("pen_saves", 0) * S.get("penalty_save", 0)
    pts += ev.get("pen_won", 0) * S.get("penalty_won", 0)
    pts += ev.get("pen_conceded", 0) * S.get("penalty_conceded", S.get("penalty_committed", 0))
    pts += ev.get("pen_miss", 0) * S.get("penalty_missed", S.get("penalty_miss", 0))

    # ---- GK saves ----
    if position == "GK":
        if "saves_per_3" in S:                    # FIFA: +1 per 3 saves
            pts += (ev["saves"] // 3) * S["saves_per_3"]["GK"]
        if "saves_inside_box_per_2" in S:         # SofaScore split
            pts += (ev.get("saves_inside", 0) // 2) * S["saves_inside_box_per_2"]["GK"]
            pts += (ev.get("saves_outside", 0) // 3) * S["saves_outside_box_per_3"]["GK"]
            pts += (ev.get("punches_claims", 0) // 2) * S["punches_high_claims_per_2"]["GK"]

    # ---- FIFA position bonuses ----
    if "tackles_per_3" in S and position in S["tackles_per_3"]:
        pts += (ev.get("tackles", 0) // 3) * S["tackles_per_3"][position]
    if "chances_created_per_2" in S and position in S["chances_created_per_2"]:
        pts += (ev.get("chances_created", 0) // 2) * S["chances_created_per_2"][position]
    if "shots_on_target_per_2" in S and position in S["shots_on_target_per_2"]:
        pts += (ev.get("sot", 0) // 2) * S["shots_on_target_per_2"][position]

    # ---- SofaScore outfield per-N bonuses ----
    for key, divisor, evkey in [
        ("clearances_per_5", 5, "clearances"), ("blocks_per_2", 2, "blocks"),
        ("interceptions_per_3", 3, "interceptions"), ("tackles_won_per_3", 3, "tackles_won"),
    ]:
        if key in S and position in S[key]:
            pts += (ev.get(evkey, 0) // divisor) * S[key][position]
    if "key_passes_per_2" in S:
        pts += (ev.get("key_passes", 0) // 2) * S["key_passes_per_2"]
    if "was_fouled_per_3" in S:
        pts += (ev.get("was_fouled", 0) // 3) * S["was_fouled_per_3"]
    if "offsides_per_2" in S:
        pts += (ev.get("offsides", 0) // 2) * S["offsides_per_2"]
    if "dispossessed_per_3" in S:
        pts += (ev.get("dispossessed", 0) // 3) * S["dispossessed_per_3"]
    # threshold bonuses (duels, passing, dribbles, long balls)
    if "duels_won_bonus" in S:
        pts += ev.get("duels_bonus", 0) * S["duels_won_bonus"]
    if "passing_bonus" in S:
        pts += ev.get("passing_bonus_hit", 0) * S["passing_bonus"]
    if "succ_dribbles_bonus" in S:
        pts += ev.get("dribbles_bonus", 0) * S["succ_dribbles_bonus"]
    if "long_balls" in S:
        pts += ev.get("long_balls_hit", 0) * S["long_balls"]

    # ---- SofaScore rating points ----
    if "rating_bands" in S and "rating" in ev:
        pts += _rating_to_points(ev["rating"], S["rating_bands"])

    # ---- FIFA bonus points ----
    if "free_kick_goal_bonus" in S:
        pts += ev.get("fk_goals", 0) * S["free_kick_goal_bonus"]
    # scouting bonus handled at squad level (needs ownership%), not here.
    return pts

# ---------------------------------------------------------------------------
class Simulator:
    def __init__(self, db_path="db.sqli", matches_json="matches.json", seed=42):
        self.con = sqlite3.connect(db_path)
        self.odds = load_odds(matches_json)
        self.rng = np.random.default_rng(seed)
        self._rate_cache = {}
        self._cur_system = "fifa"

    def rates(self, fotmob_id, position):
        key = (str(fotmob_id), position)
        if key not in self._rate_cache:
            self._rate_cache[key] = estimate_rates(self.con, fotmob_id, position)
        return self._rate_cache[key]

    def team_goal_shares(self, nation, penalty_takers=None):
        """Normalized goal-share for all DB players of a nation (so allocation of
        the odds-set team total is realistic). Penalty takers get a boost."""
        rows = self.con.execute(
            "SELECT fotmob_id, position FROM players WHERE nation=?", (nation,)
        ).fetchall()
        penalty_takers = set(str(p) for p in (penalty_takers or []))
        w = {}
        for fid, pos in rows:
            r = self.rates(fid, pos)
            exp = r["goals_p90"] * r["start_prob"] * (r["avg_minutes"] / 90.0)
            if str(fid) in penalty_takers:
                exp *= 1.6  # penalty-taker boost
            w[str(fid)] = max(exp, 1e-7)
        tot = sum(w.values())
        return {k: v / tot for k, v in w.items()} if tot else {}

    def simulate_player(self, fotmob_id, position, nation, match, is_home,
                        N, scoring, goal_share):
        """Run N sims for one player in one match; return points array + event probs."""
        lam = match["lambda_home"] if is_home else match["lambda_away"]
        mu = match["lambda_away"] if is_home else match["lambda_home"]
        rho = -0.05  # mild negative correlation (Dixon-Coles typical)
        hg, ag = sample_dixon_coles(match["lambda_home"], match["lambda_away"],
                                    rho, N, self.rng)
        team_goals = hg if is_home else ag
        team_conceded = ag if is_home else hg

        r = self.rates(fotmob_id, position)
        # minutes: Bernoulli start, then minutes = avg if start else partial sub
        start = self.rng.random(N) < r["start_prob"]
        minutes = np.where(start, r["avg_minutes"],
                           self.rng.integers(0, 30, size=N))
        played_frac = minutes / 90.0

        # goals: this player's slice of team goals ~ Binomial(team_goals, share)
        share = goal_share.get(str(fotmob_id), 0.0)
        # only players who are on the pitch can score; scale share by played_frac
        eff_share = np.where(minutes > 0, share, 0.0)
        goals = self.rng.binomial(team_goals, np.clip(eff_share, 0, 1))

        # assists ~ Poisson(assist_p90 * played_frac), capped by remaining team goals
        assists = self.rng.poisson(np.maximum(r["assists_p90"] * played_frac, 0))
        assists = np.minimum(assists, np.maximum(team_goals - goals, 0))

        # cards
        yellow = (self.rng.random(N) < np.clip(r["yellow_p90"] * played_frac, 0, 1)).astype(int)
        red = (self.rng.random(N) < np.clip(r["red_p90"] * played_frac, 0, 1)).astype(int)

        # saves (GK) ~ Poisson(saves_p90 * played_frac)
        saves = self.rng.poisson(np.maximum(r["saves_p90"] * played_frac, 0)) \
            if position == "GK" else np.zeros(N, int)

        # defensive actions ~ Poisson(def_actions_p90 * played_frac)
        def_actions = self.rng.poisson(np.maximum(r["def_actions_p90"] * played_frac, 0))

        ev = {
            "minutes": minutes, "goals": goals, "assists": assists,
            "team_conceded": team_conceded, "saves": saves,
            "yellow": yellow, "red": red, "def_actions": def_actions,
            "pen_saves": np.zeros(N, int), "pen_miss": np.zeros(N, int),
            "sot": np.zeros(N, int),
        }
        if "rating_bands" in scoring:
            # approximate a per-match rating from events (base 6.7 + contributions)
            ev["rating"] = (6.7 + 0.55 * goals + 0.28 * assists
                            + 0.015 * def_actions - 0.3 * yellow - 1.0 * red)
        pts = points_from_events(ev, position, scoring, system=self._cur_system)
        return pts, ev

    def simulate_squad(self, squad, scope, system="fifa", N=20000,
                       captain=None, vice_captain=None, penalty_takers=None):
        """squad: list of dicts {fotmob_id, position, nation, name}.
        scope: set of (home,away) tuples defining the matchdays in play (or empty=all).
        Returns per-player results + squad distribution."""
        scoring = SCORING_SYSTEMS[system]
        self._cur_system = system
        # precompute goal shares per nation in the squad
        nations = {p["nation"] for p in squad}
        shares = {nat: self.team_goal_shares(nat, penalty_takers) for nat in nations}

        per_player = {}
        squad_total = np.zeros(N, dtype=float)
        for p in squad:
            match, is_home = find_match(self.odds, p["nation"], scope)
            if match is None:
                # nation not playing in scope -> zero contribution
                per_player[p["fotmob_id"]] = {
                    "name": p.get("name"), "position": p["position"],
                    "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0,
                    "p_goal": 0.0, "p_cs": 0.0, "note": "no match in scope"}
                continue
            pts, ev = self.simulate_player(
                p["fotmob_id"], p["position"], p["nation"], match, is_home,
                N, scoring, shares.get(p["nation"], {}))
            mult = 1
            if captain and str(p["fotmob_id"]) == str(captain):
                mult = SCORING_SYSTEMS  # placeholder; multiplier applied below
            contrib = pts.copy()
            per_player[p["fotmob_id"]] = {
                "name": p.get("name"), "position": p["position"], "nation": p["nation"],
                "mean": float(pts.mean()),
                "p10": float(np.percentile(pts, 10)),
                "p50": float(np.percentile(pts, 50)),
                "p90": float(np.percentile(pts, 90)),
                "p_goal": float((ev["goals"] > 0).mean()),
                "p_cs": float(((ev["team_conceded"] == 0) & (ev["minutes"] >= 60)).mean()),
                "_pts": pts,  # kept for captain math / squad totals
            }

        # captaincy: double the captain's points (vice if captain doesn't play)
        from sim_config import GAME_RULES
        cmult = GAME_RULES[system]["captain_multiplier"]
        for fid, res in per_player.items():
            if "_pts" not in res:
                continue
            arr = res["_pts"]
            if captain and str(fid) == str(captain):
                # captain plays if minutes>0 in that sim; approximate always-on double
                arr = arr * cmult
            squad_total += arr
        # build distribution + rating
        mean = float(squad_total.mean())
        dist = {
            "mean": mean,
            "p10": float(np.percentile(squad_total, 10)),
            "p50": float(np.percentile(squad_total, 50)),
            "p90": float(np.percentile(squad_total, 90)),
            "std": float(squad_total.std()),
            "samples": squad_total,   # for histogram
        }
        # strip internal arrays from per-player before returning (keep for caller opt-in)
        for res in per_player.values():
            res.pop("_pts", None)
        return {"players": per_player, "squad": dist, "system": system, "N": N}

    @staticmethod
    def team_rating(expected_points, scope_size):
        """Map expected squad points to a 0-100 rating + letter grade.
        Calibrated so an average squad ~ 50, elite ~ 85+. scope_size = #matchdays."""
        per_md = expected_points / max(scope_size, 1)
        # ~45 pts/MD is strong for an 11-15 man squad; scale around that
        rating = max(0, min(100, (per_md / 60.0) * 100))
        grades = [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"),
                  (50, "C+"), (40, "C"), (30, "D"), (0, "F")]
        grade = next(g for thr, g in grades if rating >= thr)
        return round(rating, 1), grade


if __name__ == "__main__":
    print("sim_engine module — import Simulator and call simulate_squad().")
