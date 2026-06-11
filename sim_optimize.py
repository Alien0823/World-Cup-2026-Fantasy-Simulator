#!/usr/bin/env python3
"""
sim_optimize.py  -  Exact optimal-squad generator (WC2026 fantasy).

Given each player's expected points (from sim_engine) plus price/position/nation,
solve an integer program for the highest-expected-points legal squad under the
selected game's rules (FIFA or SofaScore): budget, squad size, position quota,
formation feasibility, max-per-nation, and captaincy (captain points doubled).

Uses PuLP (CBC solver) for an exact, guaranteed-optimal solution.

Captaincy: the optimizer chooses the captain as part of the optimization — the
captain's expected points are doubled (vice-captain handled as a fallback flag).

Wildcard awareness: a "wildcard" simply means no transfer constraints, i.e. a
fresh optimal squad — which is exactly what this solver returns. For non-wildcard
rounds you can pass an existing squad + free_transfers to limit changes.

Usage (programmatic):
    from sim_optimize import optimize_squad
    result = optimize_squad(players, system="fifa")
  where players = [{fotmob_id, name, position, nation, price, xpts}, ...]
"""
import pulp
from sim_config import GAME_RULES

def optimize_squad(players, system="fifa", budget=None,
                   locked_in=None, locked_out=None,
                   existing_squad=None, free_transfers=None,
                   transfer_penalty=4.0):
    """
    players: list of dicts with keys:
        fotmob_id, name, position (GK/DEF/MID/FWD), nation, price, xpts
    system: 'fifa' or 'sofascore'
    budget: override the rule default if given
    locked_in:  set of fotmob_ids that MUST be in the squad
    locked_out: set of fotmob_ids that must NOT be selected
    existing_squad / free_transfers: if provided, limit changes (non-wildcard).
    transfer_penalty: expected-points cost per transfer beyond free_transfers.

    Returns dict: {squad, captain, vice_captain, starting_xi, bench,
                   total_xpts, total_price, formation, status}
    """
    R = GAME_RULES[system]
    budget = budget if budget is not None else R["budget"]
    squad_size = R["squad_size"]
    pos_quota = R["positions"]
    fmin, fmax = R["formation_min"], R["formation_max"]
    max_nat = R["max_per_nation"]
    cap_mult = R["captain_multiplier"]
    has_bench = squad_size > R["starting_xi"]

    locked_in = set(str(x) for x in (locked_in or []))
    locked_out = set(str(x) for x in (locked_out or []))

    # filter out locked_out and players with no price/xpts
    P = [p for p in players
         if str(p["fotmob_id"]) not in locked_out
         and p.get("price") is not None and p.get("xpts") is not None]
    ids = [str(p["fotmob_id"]) for p in P]
    by_id = {str(p["fotmob_id"]): p for p in P}

    prob = pulp.LpProblem("wc_fantasy_squad", pulp.LpMaximize)

    # decision vars
    pick = {i: pulp.LpVariable(f"pick_{i}", cat="Binary") for i in ids}        # in squad
    start = {i: pulp.LpVariable(f"start_{i}", cat="Binary") for i in ids}      # in XI
    capt = {i: pulp.LpVariable(f"capt_{i}", cat="Binary") for i in ids}        # captain

    # objective: starters get full xpts, bench gets 0 (FIFA scores only the XI);
    # captain gets an EXTRA (cap_mult-1)*xpts on top.
    # If the game has no bench (SofaScore XI-only), start == pick.
    obj = []
    for i in ids:
        xp = by_id[i]["xpts"]
        obj.append(xp * start[i])
        obj.append(xp * (cap_mult - 1) * capt[i])
    # transfer penalty (non-wildcard)
    transfer_terms = []
    if existing_squad is not None and free_transfers is not None:
        existing = set(str(x) for x in existing_squad)
        # number transferred IN = picks not in existing
        n_in = pulp.lpSum(pick[i] for i in ids if i not in existing)
        # penalty applies to transfers beyond the free allowance
        over = pulp.LpVariable("transfers_over", lowBound=0)
        prob += over >= n_in - free_transfers
        transfer_terms.append(-transfer_penalty * over)
    prob += pulp.lpSum(obj) + pulp.lpSum(transfer_terms)

    # ---- constraints ----
    # squad size
    prob += pulp.lpSum(pick[i] for i in ids) == squad_size
    # budget
    prob += pulp.lpSum(by_id[i]["price"] * pick[i] for i in ids) <= budget
    # position quota in the SQUAD
    for pos, q in pos_quota.items():
        prob += pulp.lpSum(pick[i] for i in ids if by_id[i]["position"] == pos) == q
    # max per nation
    nations = {by_id[i]["nation"] for i in ids}
    for nat in nations:
        prob += pulp.lpSum(pick[i] for i in ids if by_id[i]["nation"] == nat) <= max_nat
    # starting XI size + formation feasibility
    if has_bench:
        prob += pulp.lpSum(start[i] for i in ids) == R["starting_xi"]
        for i in ids:
            prob += start[i] <= pick[i]           # can only start if picked
        for pos in ("GK", "DEF", "MID", "FWD"):
            s = pulp.lpSum(start[i] for i in ids if by_id[i]["position"] == pos)
            prob += s >= fmin[pos]
            prob += s <= fmax[pos]
    else:
        # XI-only game: starters are exactly the squad
        for i in ids:
            prob += start[i] == pick[i]
    # captain: exactly one, must be a starter
    prob += pulp.lpSum(capt[i] for i in ids) == 1
    for i in ids:
        prob += capt[i] <= start[i]
    # locked-in players
    for i in locked_in:
        if i in pick:
            prob += pick[i] == 1

    # solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"status": status, "squad": [], "total_xpts": 0}

    chosen = [i for i in ids if pick[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    bench = [i for i in chosen if i not in starters]
    captain = next((i for i in ids if capt[i].value() > 0.5), None)
    # vice = highest-xpts starter that isn't captain
    vice = max((i for i in starters if i != captain),
               key=lambda i: by_id[i]["xpts"], default=None)

    def info(i):
        p = by_id[i]
        return {"fotmob_id": i, "name": p.get("name"), "position": p["position"],
                "nation": p["nation"], "price": p["price"], "xpts": p["xpts"]}

    # formation string from starters
    fc = {pos: sum(1 for i in starters if by_id[i]["position"] == pos)
          for pos in ("GK", "DEF", "MID", "FWD")}
    formation = f"{fc['DEF']}-{fc['MID']}-{fc['FWD']}"

    total_xpts = sum(by_id[i]["xpts"] for i in starters)
    if captain:
        total_xpts += by_id[captain]["xpts"] * (cap_mult - 1)
    total_price = sum(by_id[i]["price"] for i in chosen)

    return {
        "status": status,
        "squad": [info(i) for i in chosen],
        "starting_xi": [info(i) for i in starters],
        "bench": [info(i) for i in bench],
        "captain": info(captain) if captain else None,
        "vice_captain": info(vice) if vice else None,
        "formation": formation,
        "total_xpts": round(total_xpts, 2),
        "total_price": round(total_price, 1),
        "budget": budget,
        "system": system,
    }


def optimize_from_simulation(sim_result, players_meta, system="fifa", **kwargs):
    """Bridge: take a sim_engine.simulate_squad() result (per-player means) +
    player metadata (price/position/nation) and run the optimizer.
    players_meta: {fotmob_id: {name, position, nation, price}}"""
    players = []
    for fid, meta in players_meta.items():
        xp = sim_result["players"].get(fid, {}).get("mean")
        if xp is None or meta.get("price") is None:
            continue
        players.append({"fotmob_id": fid, "name": meta.get("name"),
                        "position": meta["position"], "nation": meta["nation"],
                        "price": meta["price"], "xpts": xp})
    return optimize_squad(players, system=system, **kwargs)


if __name__ == "__main__":
    # self-test with a small synthetic player pool
    import random
    random.seed(1)
    nations = ["Spain", "Germany", "France", "Brazil", "England", "Norway",
               "Portugal", "Argentina", "Netherlands", "Croatia"]
    pool = []
    pid = 0
    for pos, n, pmax in [("GK", 12, 6), ("DEF", 30, 7.5), ("MID", 30, 11), ("FWD", 20, 12)]:
        for _ in range(n):
            pid += 1
            price = round(random.uniform(4.0, pmax), 1)
            xp = round(random.uniform(1, 9) * (price / pmax + 0.4), 2)
            pool.append({"fotmob_id": str(pid), "name": f"{pos}{pid}",
                         "position": pos, "nation": random.choice(nations),
                         "price": price, "xpts": xp})
    for sysname in ("fifa", "sofascore"):
        r = optimize_squad(pool, system=sysname)
        print(f"\n=== {sysname.upper()} optimal squad ({r['status']}) ===")
        print(f"formation {r['formation']}  xPts {r['total_xpts']}  cost {r['total_price']}/{r['budget']}")
        print(f"captain: {r['captain']['name'] if r['captain'] else None}  "
              f"vice: {r['vice_captain']['name'] if r['vice_captain'] else None}")
        print("XI:", ", ".join(f"{p['name']}({p['position']},{p['nation'][:3]},{p['price']})"
                               for p in r["starting_xi"]))
        if r["bench"]:
            print("bench:", ", ".join(p["name"] for p in r["bench"]))
