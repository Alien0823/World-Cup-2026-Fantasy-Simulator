#!/usr/bin/env python3
"""
sim_config.py  -  Scoring systems, competition weights, and tunable constants
for the WC2026 fantasy simulation engine.

Two scoring systems are defined (FIFA Fantasy and SofaScore Fantasy). Each maps
a player's simulated per-match events -> fantasy points, by position.

NOTE: SofaScore Fantasy scoring is approximate and centralised here so it is easy
to correct in one place once confirmed against the live game.
"""

# ---------------------------------------------------------------------------
# SCORING SYSTEMS
# Positions: GK, DEF, MID, FWD
# Each entry is points awarded for the event (per unit unless noted).
# ---------------------------------------------------------------------------
FIFA_SCORING = {
    # Appearance: +1 for any minutes, +1 again for 60+  (rules: +1 up to 60, +1 for 60+)
    # We model: 1 pt if played 1-59, 2 pts if 60+.
    "appearance_any": 1,
    "appearance_60_bonus": 1,        # extra +1 on top if 60+
    "goal": {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5},
    "assist": 3,
    "clean_sheet": {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0},  # needs 60+ mins
    # goals conceded: first conceded = 0, each ADDITIONAL = -1 (GK/DEF only)
    "conceded_after_first": {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "saves_per_3": {"GK": 1},
    "penalty_save": 3,               # not shootouts
    "penalty_won": 2,
    "penalty_conceded": -1,
    "yellow_card": -1,
    "red_card": -2,
    "own_goal": -2,
    # MID: every 3 tackles +1 ; every 2 chances created +1
    "tackles_per_3": {"MID": 1},
    "chances_created_per_2": {"MID": 1},
    # FWD: every 2 shots on target +1
    "shots_on_target_per_2": {"FWD": 1},
    # Bonus points
    "free_kick_goal_bonus": 1,       # direct free-kick goal, on top of goal pts
    "scouting_bonus": 2,             # >4 pts in match AND <5% ownership
    "scouting_pts_threshold": 4,
    "scouting_ownership_threshold": 5.0,
}

SOFASCORE_SCORING = {
    "appearance_any": 1,             # 1 pt up to 60 min
    "appearance_60_bonus": 1,        # +1 more if 60+ (total 2)
    "goal": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": {"GK": 4, "DEF": 4, "MID": 3, "FWD": 3},
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 0, "FWD": 0},  # needs 60+ mins
    "conceded_per_2": {"GK": -1, "DEF": -1, "MID": 0, "FWD": 0},  # -1 per 2 conceded
    "own_goal": -2,
    "yellow_card": -1,
    # second yellow & red are minute-dependent (earlier = worse)
    "second_yellow_by_minute": [((0, 29), -3), ((30, 59), -2), ((60, 120), -1)],
    "red_card_by_minute": [((0, 29), -4), ((30, 59), -3), ((60, 120), -2)],
    "penalty_save": 5,
    "penalty_won": 2,
    "penalty_committed": -2,
    "penalty_missed": -3,
    # GK specifics
    "saves_inside_box_per_2": {"GK": 1},
    "saves_outside_box_per_3": {"GK": 1},
    "punches_high_claims_per_2": {"GK": 1},
    "successful_runs_out": {"GK": 1},      # per the rules (1 pt)
    "long_balls": 1,        # >=3 accurate long balls, >=60% success -> +1 (all pos)
    "clearance_off_line": 2,
    # outfield defensive/attacking per-N bonuses (DEF/MID/FWD)
    "clearances_per_5": {"DEF": 1, "MID": 1, "FWD": 1},
    "blocks_per_2": {"DEF": 1, "MID": 1, "FWD": 1},
    "interceptions_per_3": {"DEF": 1, "MID": 1, "FWD": 1},
    "tackles_won_per_3": {"DEF": 1, "MID": 1, "FWD": 1},
    "duels_won_bonus": 1,   # >=3 duels won AND >=50% -> +1
    "was_fouled_per_3": 1,  # all positions
    "passing_bonus": 1,     # >=40 passes AND >=90% -> +1
    "offsides_per_2": -1,
    "dispossessed_per_3": -1,
    "key_passes_per_2": 1,
    "succ_dribbles_bonus": 1,   # >=3 successful dribbles AND >=60% -> +1
    # Sofascore Rating -> points, -2..+3.  CONFIRMED from official scale image.
    # Format: list of ((low, high), points), inclusive low, exclusive high.
    "rating_bands": [
        ((0.0, 6.0),  -2),
        ((6.0, 6.5),  -1),
        ((6.5, 7.0),   0),
        ((7.0, 8.0),   1),
        ((8.0, 9.0),   2),
        ((9.0, 10.01), 3),
    ],
    "_rating_bands_confirmed": True,    # confirmed from user's scale image
}

SCORING_SYSTEMS = {"fifa": FIFA_SCORING, "sofascore": SOFASCORE_SCORING}

# ---------------------------------------------------------------------------
# GAME CONSTRAINTS for the optimizer (squad building)
# ---------------------------------------------------------------------------
FIFA_RULES = {
    "budget": 100.0,
    "squad_size": 15,
    "starting_xi": 11,
    "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
    "formation_min": {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1},
    "formation_max": {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3},
    "allowed_formations": ["4-4-2", "4-3-3", "4-5-1", "3-4-3",
                            "3-5-2", "5-4-1", "5-3-2"],
    "max_per_nation": 3,             # group stage; rises in KO rounds
    "captain_multiplier": 2,
    "vice_captain_multiplier": 2,    # vice scores double if captain DNP
    "prices_fixed": True,            # FIFA prices don't change
    # transfers by stage (group): unlimited pre-tournament, then 2/round
    "free_transfers": {"pre": None, "md2": 2, "md3": 2},
    "transfer_hit": -3,              # per extra transfer beyond allowance
    "tokens": {
        "wildcard": "unlimited transfers one round (not MD1 or Ro32)",
        "12th_man": "an extra player scores for one round",
        "max_captain": "double points from your best starter that round",
        "qualification_booster": "+2 to XI players who progress (Ro32+)",
        "mystery": "revealed at Ro32",
    },
}

SOFASCORE_RULES = {
    "budget": 100.0,
    "squad_size": 15,
    "starting_xi": 11,
    "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
    "formation_min": {"GK": 1, "DEF": 3, "MID": 3, "FWD": 1},  # SofaScore: 3+ MID
    "formation_max": {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3},
    "max_per_nation": 3,             # group stage; rises in KO rounds
    "captain_multiplier": 2,
    "vice_captain_multiplier": 1,    # SofaScore: no auto-vice doubling described
    "prices_fixed": False,           # SofaScore prices change with form/demand
    "free_transfers": {"pre": None, "group": 3},  # 3 per round in group stage
    "transfer_carryover": 1,         # one transfer can carry over in group stage
    "tokens": {"triple_captain": "captain scores 3x instead of 2x (one use)"},
}

GAME_RULES = {"fifa": FIFA_RULES, "sofascore": SOFASCORE_RULES}

# ---------------------------------------------------------------------------
# COMPETITION WEIGHTS  (how much a match counts toward a player's rate estimate)
# Higher tier = more predictive of WC performance. Friendlies heavily discounted.
# Keyed by substrings matched (case-insensitive) against FotMob leagueName.
# ---------------------------------------------------------------------------
COMPETITION_WEIGHTS = [
    (("world cup", "fifa world cup"), 1.50),
    (("euro", "european championship", "copa america", "copa américa"), 1.40),
    (("champions league",), 1.30),
    (("europa league", "nations league a"), 1.15),
    (("nations league",), 1.05),
    (("premier league", "laliga", "la liga", "serie a", "bundesliga", "ligue 1"), 1.20),
    (("eredivisie", "primeira liga", "championship", "mls"), 1.00),
    (("world cup qualification", "wc qualification", "qualification"), 1.00),
    (("euro qualification",), 0.95),
    (("fa cup", "efl cup", "dfb pokal", "copa del rey", "coppa italia"), 0.90),
    (("friendly", "international friendly", "club friendly"), 0.45),  # heavy discount
]
DEFAULT_COMP_WEIGHT = 0.85

def competition_weight(league_name: str) -> float:
    if not league_name:
        return DEFAULT_COMP_WEIGHT
    ln = league_name.lower()
    for subs, w in COMPETITION_WEIGHTS:
        if any(s in ln for s in subs):
            return w
    return DEFAULT_COMP_WEIGHT

# ---------------------------------------------------------------------------
# FORM WEIGHTING: exponential decay by recency (match index, 0 = most recent)
# weight = decay ** index  (older matches matter less)
# ---------------------------------------------------------------------------
FORM_DECAY = 0.97          # per match step
FORM_HALFLIFE_MATCHES = 23 # ~ decay 0.97 -> half weight after ~23 matches

# ---------------------------------------------------------------------------
# BAYESIAN SHRINKAGE: blend player's raw per-90 rate toward the position mean.
# shrunk = (n*raw + K*prior) / (n + K), where n = weighted matches, K = strength.
# ---------------------------------------------------------------------------
SHRINKAGE_K = {           # pseudo-matches of prior; higher = more shrinkage
    "goals": 8, "assists": 8, "saves": 5, "def_actions": 6,
    "cards": 10, "minutes": 4,
}

# Position prior per-90 rates (rough population means; used as shrinkage target).
POSITION_PRIORS = {
    "GK":  {"goals": 0.00, "assists": 0.01, "saves": 3.0, "def_actions": 2.0,
            "yellow": 0.06, "red": 0.004},
    "DEF": {"goals": 0.06, "assists": 0.06, "saves": 0.0, "def_actions": 9.0,
            "yellow": 0.16, "red": 0.01},
    "MID": {"goals": 0.13, "assists": 0.13, "saves": 0.0, "def_actions": 6.0,
            "yellow": 0.13, "red": 0.006},
    "FWD": {"goals": 0.38, "assists": 0.13, "saves": 0.0, "def_actions": 2.5,
            "yellow": 0.10, "red": 0.005},
}

# Opponent-strength weighting: matches vs strong teams count slightly more when
# estimating attacking output (scoring vs a good side is more predictive).
# Scale by opponent FotMob-rating or odds-implied strength if available; default 1.0.
OPPONENT_STRENGTH_RANGE = (0.85, 1.20)
