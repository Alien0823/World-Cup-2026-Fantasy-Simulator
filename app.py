#!/usr/bin/env python3
"""
app.py  -  WC2026 Fantasy Simulator GUI (Streamlit)

Run locally:
    pip install streamlit numpy scipy pulp
    streamlit run app.py

Reads:
    db.sqli          (players + detailed match stats, from the scraper pipeline)
    matches.json   (de-vigged odds / lambdas per fixture)

Features:
    - Search + filter the full player pool, build a 15-man squad
    - Choose scoring system (FIFA / SofaScore), with a one-click compare toggle
    - Choose matchday scope (MD1 / MD1+2 / MD1+2+3) and #simulations
    - Per-player expected points, P(goal), P(clean sheet)
    - Squad points distribution chart + team rating & grade
    - One-click optimal-squad generator (exact ILP)
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import streamlit as st

from sim_engine import Simulator
from sim_optimize import optimize_squad
from sim_config import GAME_RULES

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="WC2026 Fantasy Simulator", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

PITCH = "#0B6E4F"; PITCH_LIGHT = "#15966B"; CHALK = "#E8EEE9"
NIGHT = "#0C1116"; PANEL = "#141C24"; ACCENT = "#F2C14E"; MUTE = "#7C8B99"

st.markdown(f"""
<style>
  .stApp {{ background: {NIGHT}; color: {CHALK}; }}
  section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid #1f2a35; }}
  h1, h2, h3 {{ font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.02em; }}
  .hero {{ font-size: 2.0rem; font-weight: 800; line-height: 1.1;
           background: linear-gradient(90deg, {CHALK}, {PITCH_LIGHT});
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .eyebrow {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 700;
              letter-spacing: 0.18em; text-transform: uppercase; }}
  .metric-card {{ background: {PANEL}; border: 1px solid #1f2a35; border-radius: 14px;
                  padding: 14px 18px; }}
  .grade {{ font-size: 2.6rem; font-weight: 800; color: {ACCENT}; line-height: 1; }}
  .pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px;
           font-size: 0.72rem; font-weight: 700; background: #1f2a35; color: {PITCH_LIGHT}; }}
  .stButton>button {{ background: {PITCH}; color: white; border: 0; border-radius: 10px;
                      font-weight: 700; padding: 0.5rem 1.1rem; }}
  .stButton>button:hover {{ background: {PITCH_LIGHT}; }}
  div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DB = "db.sqli"
ODDS = "matches.json"

@st.cache_data(show_spinner=False)
def load_players():
    if not Path(DB).exists():
        return []
    con = sqlite3.connect(DB)
    rows = con.execute("""SELECT fotmob_id, name, nation, position, club,
                          price_fifa, price_sofascore FROM players
                          ORDER BY nation, position""").fetchall()
    con.close()
    cols = ["fotmob_id", "name", "nation", "position", "club",
            "price_fifa", "price_sofascore"]
    return [dict(zip(cols, r)) for r in rows]

@st.cache_data(show_spinner=False)
def load_fixtures():
    if not Path(ODDS).exists():
        return []
    data = json.loads(Path(ODDS).read_text(encoding="utf-8"))
    return data["matches"]

players = load_players()
fixtures = load_fixtures()
POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">World Cup 2026 · Fantasy Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Squad Simulator & Optimiser</div>', unsafe_allow_html=True)
st.caption("Monte-Carlo fantasy projections from bookmaker odds + FotMob form. "
           "Pick a squad, simulate, and let the optimiser find the best XI.")

if not players:
    st.warning("No `db.sqli` found yet. Run the scraper pipeline first to build the "
               "player database, then reload this page.")
    st.stop()
if not fixtures:
    st.warning("No `matches.json` found. Run `parse_odds.py` to generate fixture odds.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    system = st.radio("Scoring system", ["fifa", "sofascore"],
                      format_func=lambda s: "FIFA Fantasy" if s == "fifa" else "SofaScore Fantasy")
    compare = st.toggle("Compare both systems", value=False)
    scope_choice = st.selectbox("Matchday scope",
                                ["MD1", "MD1 + MD2", "MD1 + MD2 + MD3"])
    n_sims = st.select_slider("Simulations", options=[2000, 5000, 10000, 20000, 50000, 100000],
                              value=20000)
    st.divider()
    st.markdown("### Squad")
    st.caption(f"Budget €{GAME_RULES[system]['budget']:.0f}M · "
               f"15 players · max 3 per nation")

price_key = "price_fifa" if system == "fifa" else "price_sofascore"
scope_n = {"MD1": 1, "MD1 + MD2": 2, "MD1 + MD2 + MD3": 3}[scope_choice]

# ---------------------------------------------------------------------------
# Player selection
# ---------------------------------------------------------------------------
st.markdown("#### 1 · Build your squad")
fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
with fcol1:
    search = st.text_input("Search players", placeholder="Name, club, or nation…")
with fcol2:
    pos_filter = st.multiselect("Position", ["GK", "DEF", "MID", "FWD"], default=[])
with fcol3:
    nat_filter = st.multiselect("Nation", sorted({p["nation"] for p in players}), default=[])

def matches_filter(p):
    if search:
        s = search.lower()
        if s not in (p["name"] or "").lower() and s not in (p["club"] or "").lower() \
           and s not in (p["nation"] or "").lower():
            return False
    if pos_filter and p["position"] not in pos_filter:
        return False
    if nat_filter and p["nation"] not in nat_filter:
        return False
    return True

filtered = [p for p in players if matches_filter(p)]
filtered.sort(key=lambda p: (POS_ORDER.get(p["position"], 9), -(p.get(price_key) or 0)))

if "squad" not in st.session_state:
    st.session_state.squad = {}   # fotmob_id -> player dict

# show filtered pool as a selectable table
st.caption(f"{len(filtered)} players match. Click + to add.")
pool_box = st.container(height=280)
with pool_box:
    for p in filtered[:120]:
        c1, c2, c3, c4, c5 = st.columns([0.5, 3, 2, 1, 1])
        price = p.get(price_key)
        c1.write(f"**{p['position']}**")
        c2.write(f"{p['name']}")
        c3.caption(f"{p['nation']} · {p['club'] or ''}")
        c4.write(f"€{price}M" if price else "—")
        if c5.button("➕", key=f"add_{p['fotmob_id']}"):
            if len(st.session_state.squad) < 15:
                st.session_state.squad[p["fotmob_id"]] = p
                st.rerun()

# ---------------------------------------------------------------------------
# Current squad panel
# ---------------------------------------------------------------------------
squad = list(st.session_state.squad.values())
total_cost = sum((p.get(price_key) or 0) for p in squad)
budget = GAME_RULES[system]["budget"]
quota = GAME_RULES[system]["positions"]
counts = {k: sum(1 for p in squad if p["position"] == k) for k in ("GK", "DEF", "MID", "FWD")}

st.markdown("#### 2 · Your squad")
mcol = st.columns(5)
mcol[0].markdown(f'<div class="metric-card"><div class="eyebrow">Players</div>'
                 f'<div style="font-size:1.6rem;font-weight:800">{len(squad)}/15</div></div>',
                 unsafe_allow_html=True)
mcol[1].markdown(f'<div class="metric-card"><div class="eyebrow">Spent</div>'
                 f'<div style="font-size:1.6rem;font-weight:800">€{total_cost:.1f}M</div>'
                 f'<div class="pill">€{budget-total_cost:.1f}M left</div></div>',
                 unsafe_allow_html=True)

# position counts in a row below
pc = st.columns(4)
for i, pos in enumerate(("GK", "DEF", "MID", "FWD")):
    ok = counts[pos] == quota[pos]
    color = PITCH_LIGHT if ok else MUTE
    pc[i].markdown(f'<div class="metric-card"><div class="eyebrow">{pos}</div>'
                   f'<div style="font-size:1.3rem;font-weight:800;color:{color}">'
                   f'{counts[pos]}/{quota[pos]}</div></div>', unsafe_allow_html=True)

if squad:
    rmcols = st.columns(3)
    for i, p in enumerate(sorted(squad, key=lambda x: POS_ORDER.get(x["position"], 9))):
        col = rmcols[i % 3]
        if col.button(f"✕ {p['position']} · {p['name']} ({p['nation']})",
                      key=f"rm_{p['fotmob_id']}"):
            del st.session_state.squad[p["fotmob_id"]]
            st.rerun()

# captain / penalty pickers
cap_col, pen_col = st.columns(2)
captain = None
if squad:
    names = {f"{p['name']} ({p['nation']})": p["fotmob_id"] for p in squad}
    cap_label = cap_col.selectbox("Captain (points doubled)", list(names.keys()))
    captain = names.get(cap_label)
    pen_labels = pen_col.multiselect("Penalty takers (goal-share boost)", list(names.keys()))
    penalty_takers = [names[l] for l in pen_labels]
else:
    penalty_takers = []

# ---------------------------------------------------------------------------
# Build scope set (which fixtures count)
# ---------------------------------------------------------------------------
def scope_fixtures(n_md):
    """Return the set of (home,away) fixtures for the first n_md matchdays.
    Robust to fixture ordering: for each nation, take its first n_md fixtures
    in file order (each nation plays once per group-stage matchday)."""
    seen_count = {}
    chosen = set()
    for m in fixtures:
        for team in (m["home"], m["away"]):
            seen_count[team] = seen_count.get(team, 0)
        # this fixture counts as matchday = max appearances of its two teams so far
        md_home = seen_count[m["home"]] + 1
        md_away = seen_count[m["away"]] + 1
        md = max(md_home, md_away)
        if md <= n_md:
            chosen.add((m["home"], m["away"]))
        seen_count[m["home"]] += 1
        seen_count[m["away"]] += 1
    return chosen

scope = scope_fixtures(scope_n)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
st.markdown("#### 3 · Simulate")
run = st.button("▶  Run simulation", type="primary", use_container_width=True)

def run_sim(system_name):
    sim = Simulator(DB, ODDS, seed=42)
    sq = [{"fotmob_id": p["fotmob_id"], "position": p["position"],
           "nation": p["nation"], "name": p["name"]} for p in squad]
    return sim.simulate_squad(sq, scope=scope, system=system_name, N=n_sims,
                              captain=captain, penalty_takers=penalty_takers)

def render_result(res, system_name):
    rating, grade = Simulator.team_rating(res["squad"]["mean"], scope_n)
    a, b, c = st.columns([1, 1, 2])
    a.markdown(f'<div class="metric-card"><div class="eyebrow">Team rating</div>'
               f'<div class="grade">{grade}</div>'
               f'<div class="pill">{rating}/100</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="metric-card"><div class="eyebrow">Expected points</div>'
               f'<div style="font-size:2.0rem;font-weight:800">{res["squad"]["mean"]:.0f}</div>'
               f'<div class="pill">p10 {res["squad"]["p10"]:.0f} · p90 {res["squad"]["p90"]:.0f}</div>'
               f'</div>', unsafe_allow_html=True)
    # distribution chart
    samples = res["squad"]["samples"]
    hist, edges = np.histogram(samples, bins=40)
    import pandas as pd
    chart_df = pd.DataFrame({"points": (edges[:-1] + edges[1:]) / 2, "freq": hist})
    c.markdown('<div class="eyebrow">Points distribution</div>', unsafe_allow_html=True)
    c.bar_chart(chart_df.set_index("points"), height=160, color=PITCH_LIGHT)

    # per-player table
    rows = []
    for fid, pr in res["players"].items():
        rows.append({"Player": pr.get("name"), "Pos": pr.get("position"),
                     "Nation": pr.get("nation", ""),
                     "xPts": round(pr["mean"], 2),
                     "P(goal)": f"{pr['p_goal']*100:.0f}%",
                     "P(clean sheet)": f"{pr['p_cs']*100:.0f}%",
                     "p10": round(pr["p10"], 1), "p90": round(pr["p90"], 1)})
    rows.sort(key=lambda r: -r["xPts"])
    st.dataframe(rows, use_container_width=True, hide_index=True)

if run:
    if len(squad) == 0:
        st.error("Add some players to your squad first.")
    else:
        with st.spinner(f"Running {n_sims:,} simulations…"):
            if compare:
                lc, rc = st.columns(2)
                with lc:
                    st.markdown("##### FIFA Fantasy")
                    render_result(run_sim("fifa"), "fifa")
                with rc:
                    st.markdown("##### SofaScore Fantasy")
                    render_result(run_sim("sofascore"), "sofascore")
            else:
                render_result(run_sim(system), system)

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
st.divider()
st.markdown("#### 4 · Optimal squad")
st.caption("Runs the simulation over the full player pool, then solves for the "
           "highest expected-points legal squad under the selected rules.")
opt_run = st.button("✦  Generate optimal squad", use_container_width=True)

if opt_run:
    with st.spinner("Simulating player pool & optimising… (this can take a moment)"):
        sim = Simulator(DB, ODDS, seed=7)
        # only players whose nation plays in scope and who have a price
        pool = [p for p in players if p.get(price_key) is not None]
        # simulate each player's xpts individually (as a 1-player squad in scope)
        xp = {}
        # group by nation to reuse goal shares
        sq_all = [{"fotmob_id": p["fotmob_id"], "position": p["position"],
                   "nation": p["nation"], "name": p["name"]} for p in pool]
        res = sim.simulate_squad(sq_all, scope=scope, system=system,
                                 N=min(n_sims, 10000))
        opt_players = []
        for p in pool:
            r = res["players"].get(p["fotmob_id"])
            if r and r["mean"] > 0:
                opt_players.append({"fotmob_id": p["fotmob_id"], "name": p["name"],
                                    "position": p["position"], "nation": p["nation"],
                                    "price": p.get(price_key), "xpts": r["mean"]})
        result = optimize_squad(opt_players, system=system)
    if result.get("status") == "Optimal":
        st.success(f"Optimal {result['formation']} · "
                   f"{result['total_xpts']} xPts · €{result['total_price']}M")
        cap = result["captain"]["name"] if result["captain"] else "—"
        vice = result["vice_captain"]["name"] if result["vice_captain"] else "—"
        st.markdown(f"**Captain:** {cap}  ·  **Vice:** {vice}")
        xi = [{"Player": p["name"], "Pos": p["position"], "Nation": p["nation"],
               "€M": p["price"], "xPts": round(p["xpts"], 2)}
              for p in result["starting_xi"]]
        xi.sort(key=lambda r: POS_ORDER.get(r["Pos"], 9))
        st.markdown("**Starting XI**")
        st.dataframe(xi, use_container_width=True, hide_index=True)
        if result["bench"]:
            bench = [{"Player": p["name"], "Pos": p["position"], "Nation": p["nation"],
                      "€M": p["price"], "xPts": round(p["xpts"], 2)} for p in result["bench"]]
            st.markdown("**Bench**")
            st.dataframe(bench, use_container_width=True, hide_index=True)
    else:
        st.error(f"Optimiser status: {result.get('status')}. "
                 "Try a wider scope or check that prices are loaded.")
