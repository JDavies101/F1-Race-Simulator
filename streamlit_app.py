# =============================================================================
# F1 RACE STRATEGY SIMULATOR — STREAMLIT WEB APP
# =============================================================================
# Interactive front-end for the Monte Carlo race strategy simulator.
# Users can adjust all race parameters via sidebar sliders, load track presets
# for all 24 rounds of the 2026 F1 calendar, run parallel simulations, and
# view interactive strategy distribution plots and lap time traces in real time.
#
# Architecture:
#   - Sidebar: all config inputs (sliders, radio, expanders, track preset loader)
#   - Run button: triggers parallel simulation via ProcessPoolExecutor
#   - st.session_state: persists results across Streamlit reruns (slider moves)
#   - Results section: metrics, strategy summary, Plotly plots, tables, download
#   - Mobile toggle: uses st.container() vs st.columns() for layout switching
#
# Plot rendering:
#   - All distribution/histogram/bar plots use Plotly for interactivity
#   - Lap traces use matplotlib for compound shading and pit spike detail
#   - Plotly plots can be downloaded via the camera icon on each chart
#   - Lap traces are downloadable as PNG via the zip download button
#
# Key Streamlit behavior:
#   Every user interaction reruns the entire script. Session state preserves
#   results across reruns so plots don't reset when sliders are adjusted.
# =============================================================================

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io
import zipfile
from collections import Counter, defaultdict
from monte_carlo_medium_v1 import simulate_race, run_simulation, generate_safety_cars
from track_presets import TRACK_PRESETS

# -----------------------------------------------------------------------------
# PAGE CONFIG
# Must be the very first Streamlit call — throws an error otherwise.
# 'wide' layout uses full browser width.
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="F1 Strategy Simulator")
st.title("F1 Race Strategy Simulator")
st.write("By Jake Davies")

# =============================================================================
# SIDEBAR: CONFIGURATION INPUTS
# =============================================================================

st.sidebar.header("Race Parameters")

# Mobile layout toggle — switches between st.columns() and st.container()
mobile_mode = st.sidebar.toggle("Mobile layout", value=False)

# --- Track Preset Loader ---
# Writes preset values into session_state and clears stale results on load.
track = st.sidebar.selectbox("Track Preset", list(TRACK_PRESETS.keys()))

if track != "Custom" and st.sidebar.button("Load Preset"):
    preset = TRACK_PRESETS[track]
    for key, val in preset.items():
        st.session_state[key] = val
    st.session_state.pop('results', None)  # clear results so stale plots don't show
    st.rerun()

# --- Main Race Parameters ---
base_lap_time = st.sidebar.slider("Base Lap Time (s)", 55, 150, st.session_state.get('base_lap_time', 80), key='base_lap_time')
total_laps    = st.sidebar.slider("Total Laps", 40, 90, st.session_state.get('total_laps', 60), key='total_laps')
pit_loss      = st.sidebar.slider("Pit Loss (s)", 15, 35, st.session_state.get('pit_loss', 25), key='pit_loss')
sc_chance_pct = st.sidebar.slider("Safety Car Chance (%)", 0, 100, st.session_state.get('sc_chance_pct', 50), key='sc_chance_pct')
n_simulations = st.sidebar.slider("Simulations", 100, 10000, st.session_state.get('n_simulations', 500), step=100, key='n_simulations')
search_method = st.sidebar.radio("Search Method", ["sampled", "exhaustive"])

# --- Compound Settings Expander ---
# Each slider has a unique key= to prevent Streamlit confusing identical labels.
with st.sidebar.expander("Compound Settings"):
    st.write("**Soft**")
    soft_offset = st.slider("Offset (s)", -3.0, 0.0, st.session_state.get('soft_offset', -1.0), step=0.1, key="soft_offset")
    soft_deg    = st.slider("Deg rate", 0.01, 0.15, st.session_state.get('soft_deg', 0.08), step=0.01, key="soft_deg")
    soft_max    = st.slider("Max laps", 5, 30, st.session_state.get('soft_max', 15), key="soft_max")
    st.write("**Medium**")
    med_offset  = st.slider("Offset (s)", -1.0, 1.0, st.session_state.get('med_offset', 0.0), step=0.1, key="med_offset")
    med_deg     = st.slider("Deg rate", 0.01, 0.15, st.session_state.get('med_deg', 0.05), step=0.01, key="med_deg")
    med_max     = st.slider("Max laps", 20, 50, st.session_state.get('med_max', 35), key="med_max")
    st.write("**Hard**")
    hard_offset = st.slider("Offset (s)", 0.0, 3.0, st.session_state.get('hard_offset', 1.5), step=0.1, key="hard_offset")
    hard_deg    = st.slider("Deg rate", 0.01, 0.10, st.session_state.get('hard_deg', 0.02), step=0.01, key="hard_deg")
    hard_max    = st.slider("Max laps", 30, 70, st.session_state.get('hard_max', 50), key="hard_max")

# --- Traffic & Safety Car Expander ---
with st.sidebar.expander("Traffic & Safety Car Settings"):
    base_traffic_prob = st.slider("Base traffic probability", 0.0, 1.0,
                                  st.session_state.get('base_traffic_prob', 0.4), step=0.05, key="base_traffic_prob")
    spread_rate       = st.slider("Traffic spread rate", 5, 40,
                                  st.session_state.get('spread_rate', 15), key="spread_rate")
    traffic_time_loss = st.slider("Traffic time loss (s)", 0.1, 2.0,
                                  st.session_state.get('traffic_time_loss', 0.5), step=0.1, key="traffic_loss")
    sc_pit_loss       = st.slider("SC pit loss (s)", 5, 20,
                                  st.session_state.get('sc_pit_loss', 12), key="sc_pit_loss")
    sc_base_lap_time  = st.slider("SC lap time (s)", 80, 120,
                                  st.session_state.get('sc_base_lap_time', 90), key="sc_lap_time")

# --- App description ---
st.markdown("""
This simulator uses **Monte Carlo methods** to find the optimal pit stop strategy for an F1 race.
Rather than a single deterministic answer, it runs thousands of randomized race scenarios to produce
a *distribution* of optimal strategies — reflecting the uncertainty real strategists face on race day.
""")

with st.expander("How it works"):
    st.markdown("""
    **The core idea**  
    Every F1 race is different. Safety cars, traffic, and tire variability mean the "optimal" pit lap 
    changes from race to race. Instead of finding one answer, this simulator asks: *across thousands 
    of random scenarios, which strategy wins most often?*
    
    **Each simulation:**
    1. Generates a random safety car schedule for the race
    2. Evaluates candidate pit strategies against that scenario
    3. Records the best 1-stop and 2-stop strategy
    
    **Tire model**  
    Each compound has a lap time offset vs the medium baseline, a degradation rate (seconds lost per lap), 
    and a cliff — the age beyond which degradation triples, modelling real grip drop-off behavior.
    
    **Traffic model**  
    After a pit stop, a car rejoins in traffic. The probability of being held up decays exponentially 
    as the field spreads out over the stint, resetting at each stop.
    
    **Safety car**  
    SC events reduce the effective pit loss — pitting under yellow is cheaper relative to cars staying out.
    
    **How to use it**  
    1. Select a track preset from the sidebar or set parameters manually  
    2. Adjust tire compounds, traffic, and SC settings if desired  
    3. Choose number of simulations (more = smoother results, slower runtime)  
    4. Hit **Run Simulation**
    """)

# =============================================================================
# CONFIG DICT
# sc_chance converts per-race % to per-lap probability:
#   sc_chance_pct / total_laps / 100
# =============================================================================
config = {
    'base_lap_time':     base_lap_time,
    'pit_loss':          pit_loss,
    'sc_pit_loss':       sc_pit_loss,
    'sc_base_lap_time':  sc_base_lap_time,
    'total_laps':        total_laps,
    'sc_chance':         sc_chance_pct / total_laps / 100,
    'base_traffic_prob': base_traffic_prob,
    'spread_rate':       spread_rate,
    'traffic_time_loss': traffic_time_loss,
    'compound_data': {
        'soft':   {'lap_time_offset': soft_offset, 'deg_rate': soft_deg, 'max_laps': soft_max},
        'medium': {'lap_time_offset': med_offset,  'deg_rate': med_deg,  'max_laps': med_max},
        'hard':   {'lap_time_offset': hard_offset, 'deg_rate': hard_deg, 'max_laps': hard_max},
    },
    'search_method':        search_method,
    'n_strategies_sampled': 500,
}

# DPI for matplotlib lap trace figures only
plot_dpi = 500

# =============================================================================
# LAP TRACE PLOT FUNCTION — Plotly version
# Returns an interactive Plotly figure with stint shading and hover tooltips.
# Kept separate from Plotly distribution plots since it requires special
# handling for vrect legend entries.
# =============================================================================

def plot_single_race_st(config, pit_laps, compounds):
    """
    Generate an interactive Plotly lap time trace for a single representative race.

    Stint regions are shaded by compound color. Pit stop laps are marked with
    vertical dashed lines. Hover shows lap number and lap time.

    Parameters
    ----------
    config : dict
    pit_laps : list of int
    compounds : list of str

    Returns
    -------
    fig : plotly Figure
    """
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
    _, lap_times = simulate_race(
        pit_laps[0], compounds, sc_laps, config,
        pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)

    compound_colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'gray'}
    stint_boundaries = [0] + pit_laps + [config['total_laps']]

    fig = go.Figure()

    # Shade each stint and add invisible scatter trace for legend entry
    for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
        color = compound_colors[compounds[i]]
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.15,
                      layer='below', line_width=0)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color=color, size=10),
            name=f'Stint {i+1}: {compounds[i][0].upper()}'
        ))

    # Pit stop vertical lines
    for pit in pit_laps:
        fig.add_vline(x=pit, line_dash='dash', line_color='red',
                      annotation_text=f'Pit lap {pit}')

    # Lap time trace — royalblue works on both light and dark backgrounds
    fig.add_trace(go.Scatter(
        x=list(range(1, config['total_laps'] + 1)),
        y=lap_times,
        mode='lines',
        name='Lap time',
        line=dict(color='royalblue', width=2),
        hovertemplate='Lap %{x}<br>%{y:.2f}s<extra></extra>'
    ))

    fig.update_layout(
        title='Single race lap time trace',
        xaxis_title='Lap',
        yaxis_title='Lap time (s)',
        yaxis_range=[config['base_lap_time'] - 2,
                     config['base_lap_time'] + config['pit_loss'] + 10]
    )
    return fig

# =============================================================================
# HELPER: stacked compound histogram (Plotly)
# =============================================================================

def make_compound_hist(soft, med, hard, title, total_laps):
    bin_spec = dict(start=1, end=total_laps, size=1)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=soft, name='S', marker_color='red',  opacity=0.8, xbins=bin_spec,
                               marker_line=dict(color='white', width=0.5)))
    fig.add_trace(go.Histogram(x=med,  name='M', marker_color='gold', opacity=0.8, xbins=bin_spec,
                               marker_line=dict(color='white', width=0.5)))
    fig.add_trace(go.Histogram(x=hard, name='H', marker_color='gray', opacity=0.8, xbins=bin_spec,
                               marker_line=dict(color='white', width=0.5)))
    fig.update_layout(barmode='stack', title=title,
                      xaxis_title='Optimal Pit Lap', yaxis_title='Frequency')
    return fig

# =============================================================================
# RUN SIMULATION BUTTON
# ProcessPoolExecutor integrates more cleanly with Streamlit than Pool.
# as_completed() allows real-time progress bar updates.
# =============================================================================

if st.button("Run Simulation"):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = [(config, sim) for sim in range(n_simulations)]
    progress_bar = st.progress(0)
    status_text  = st.empty()

    sim_results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_simulation, arg): i for i, arg in enumerate(args)}
        for i, future in enumerate(as_completed(futures)):
            sim_results.append(future.result())
            progress_bar.progress((i + 1) / n_simulations)
            status_text.text(f"Simulation {i+1}/{n_simulations}")

    st.session_state['results'] = sim_results
    st.session_state['config']  = config
    st.success(f"Done! {n_simulations} simulations complete.")

# =============================================================================
# RESULTS SECTION
# Renders only when results exist in session_state.
# Uses stored config so plots always match the simulation that was run.
# =============================================================================

if 'results' in st.session_state:
    with st.spinner("Preparing Results..."):

        sim_results  = st.session_state['results']
        config       = st.session_state['config']
        plot_buffers = {}  # stores matplotlib lap trace PNGs for download

        # --- Unpack results ---
        optimal_laps              = []
        optimal_laps_1_stop       = []
        optimal_laps_2_stop_pit1  = []
        optimal_laps_2_stop_pit2  = []
        optimal_compounds_1_stop  = []
        optimal_compounds_2_stop  = []
        best_times_1_stop         = []
        best_times_2_stop         = []
        winning_strategies_1_stop = []  # only when 1-stop beat 2-stop overall
        winning_strategies_2_stop = []  # only when 2-stop beat 1-stop overall

        for best_time_1, best_time_2, best_1, best_2, _, _ in sim_results:
            pit1, c1         = best_1
            pit2a, pit2b, c2 = best_2
            best = min([(best_time_1, '1-stop', best_1),
                        (best_time_2, '2-stop', best_2)])
            optimal_laps.append(best[1])
            optimal_laps_1_stop.append(pit1)
            optimal_laps_2_stop_pit1.append(pit2a)
            optimal_laps_2_stop_pit2.append(pit2b)
            optimal_compounds_1_stop.append(c1)
            optimal_compounds_2_stop.append(c2)
            best_times_1_stop.append(best_time_1)
            best_times_2_stop.append(best_time_2)
            if best[1] == '1-stop':
                winning_strategies_1_stop.append((pit1, c1, best_time_1))
            else:
                winning_strategies_2_stop.append((pit2a, pit2b, c2, best_time_2))

        most_common_strategy = max(set(optimal_laps), key=optimal_laps.count)
        avg_time_1stop = np.mean(best_times_1_stop)
        avg_time_2stop = np.mean(best_times_2_stop)

        # --- Representative strategies ---
        # Pulled from rank 1 of winning strategy counters so lap traces
        # always match the strategy tables exactly.
        times_by_strategy_1 = defaultdict(list)
        times_by_strategy_2 = defaultdict(list)
        for pit, comp, t in winning_strategies_1_stop:
            times_by_strategy_1[(pit, comp)].append(t)
        for pit1, pit2, comp, t in winning_strategies_2_stop:
            times_by_strategy_2[(pit1, pit2, comp)].append(t)
        pit_lap_counts_1 = Counter({k: len(v) for k, v in times_by_strategy_1.items()})
        pit_lap_counts_2 = Counter({k: len(v) for k, v in times_by_strategy_2.items()})

        if pit_lap_counts_1:
            top_freq = pit_lap_counts_1.most_common(1)[0][1]
            # among all strategies with top frequency, pick lowest avg time
            top_candidates = [k for k, v in pit_lap_counts_1.items() if v == top_freq]
            top_1stop = min(top_candidates, key=lambda k: np.mean(times_by_strategy_1[k]))
            best_1stop_pit       = top_1stop[0]
            best_1stop_compounds = list(top_1stop[1])
        else:
            best_1stop_pit       = config['total_laps'] // 2
            best_1stop_compounds = list(Counter(optimal_compounds_1_stop).most_common(1)[0][0])

        if pit_lap_counts_2:
            top_freq = pit_lap_counts_2.most_common(1)[0][1]
            top_candidates = [k for k, v in pit_lap_counts_2.items() if v == top_freq]
            top_2stop = min(top_candidates, key=lambda k: np.mean(times_by_strategy_2[k]))
            best_2stop_pit1      = top_2stop[0]
            best_2stop_pit2      = top_2stop[1]
            best_2stop_compounds = list(top_2stop[2])
        else:
            best_2stop_pit1      = config['total_laps'] // 3
            best_2stop_pit2      = 2 * config['total_laps'] // 3
            best_2stop_compounds = list(Counter(optimal_compounds_2_stop).most_common(1)[0][0])

        # =========================================================================
        # METRICS ROW
        # =========================================================================
        if mobile_mode:
            st.write("Most Common Strategy", most_common_strategy)
            st.write("1-Stop Win Rate", f"{optimal_laps.count('1-stop')}/{n_simulations}")
            st.write("2-Stop Win Rate", f"{optimal_laps.count('2-stop')}/{n_simulations}")
            st.write("Avg Best 1-Stop Time", f"{avg_time_1stop:.1f}s")
            st.write("Avg Best 2-Stop Time", f"{avg_time_2stop:.1f}s")
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Most Common Strategy",  most_common_strategy)
            col2.metric("1-Stop Win Rate",        f"{optimal_laps.count('1-stop')}/{n_simulations}")
            col3.metric("2-Stop Win Rate",        f"{optimal_laps.count('2-stop')}/{n_simulations}")
            col4.metric("Avg Best 1-Stop Time",   f"{avg_time_1stop:.1f}s")
            col5.metric("Avg Best 2-Stop Time",   f"{avg_time_2stop:.1f}s")

        # =========================================================================
        # STRATEGY SUMMARY
        # =========================================================================
        st.subheader("Strategy Summary")
        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Best 1-Stop:** Pit lap {best_1stop_pit}, compounds {' → '.join([c[0].upper() for c in best_1stop_compounds])}")
            st.write(f"**Avg race time:** {avg_time_1stop:.1f}s ({avg_time_1stop//60:.0f}m {avg_time_1stop%60:.1f}s)")
        with c2:
            st.write(f"**Best 2-Stop:** Pit laps {best_2stop_pit1}/{best_2stop_pit2}, compounds {' → '.join([c[0].upper() for c in best_2stop_compounds])}")
            st.write(f"**Avg race time:** {avg_time_2stop:.1f}s ({avg_time_2stop//60:.0f}m {avg_time_2stop%60:.1f}s)")

        # =========================================================================
        # STRATEGY DISTRIBUTION PLOT — centered on desktop
        # =========================================================================
        if mobile_mode:
            center = st.container()
        else:
            _, center, _ = st.columns([1, 2, 1])
        with center:
            st.subheader("Strategy Distribution")
            counts = Counter(optimal_laps)
            fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                         labels={'x': 'Strategy', 'y': 'Frequency'},
                         title=f"Best Pit Strategy Distribution ({n_simulations:,} simulations)")
            st.plotly_chart(fig, width='stretch')

        # =========================================================================
        # PIT LAP DISTRIBUTION PLOTS
        # Vertical lines show the deterministic optimum (equal stint split).
        # =========================================================================
        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.subheader("1-Stop Pit Lap Distribution")
            fig = px.histogram(x=optimal_laps_1_stop, nbins=config['total_laps'],
                   labels={'x': 'Optimal Pit Lap'},
                   title=f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
            fig.add_vline(x=config['total_laps']/2, line_dash='dash', line_color='red',
                        annotation_text='Deterministic optimum')
            fig.update_traces(marker_line=dict(color='white', width=0.5))
            fig.update_layout(xaxis_title='Optimal Pit Lap', yaxis_title='Frequency')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("2-Stop Pit Lap Distribution")
            bin_start = 1
            bin_end = config['total_laps']
            bin_size = 1  # one bin per lap
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=optimal_laps_2_stop_pit1, name='First stop',
                                    opacity=0.7,
                                    xbins=dict(start=bin_start, end=bin_end, size=bin_size),
                                    marker_line=dict(color='white', width=0.5)))
            fig.add_trace(go.Histogram(x=optimal_laps_2_stop_pit2, name='Second stop',
                                    opacity=0.7,
                                    xbins=dict(start=bin_start, end=bin_end, size=bin_size),
                                    marker_line=dict(color='white', width=0.5)))
            fig.update_layout(barmode='overlay',
                              title=f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)",
                              xaxis_title='Optimal Pit Lap', yaxis_title='Frequency')
            fig.add_vline(x=config['total_laps']/3,   line_dash='dash', line_color='red',
                          annotation_text='Stop 1 optimum')
            fig.add_vline(x=2*config['total_laps']/3, line_dash='dash', line_color='orange',
                          annotation_text='Stop 2 optimum')
            st.plotly_chart(fig, width='stretch')

        # =========================================================================
        # PIT LAP BY COMPOUND — stacked histograms
        # Explains why pit lap distributions are often bimodal.
        # =========================================================================
        soft_1stop = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'soft']
        med_1stop  = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'medium']
        hard_1stop = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'hard']
        soft_pit1  = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'soft']
        med_pit1   = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'medium']
        hard_pit1  = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'hard']
        soft_pit2  = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'soft']
        med_pit2   = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'medium']
        hard_pit2  = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'hard']

        st.subheader("Pit Lap Distribution by Compound")
        if mobile_mode:
            c1, c2, c3 = st.container(), st.container(), st.container()
        else:
            c1, c2, c3 = st.columns(3)

        with c1:
            st.plotly_chart(make_compound_hist(soft_1stop, med_1stop, hard_1stop,
                            "1-Stop — opening compound", config['total_laps']),
                            width='stretch')
        with c2:
            st.plotly_chart(make_compound_hist(soft_pit1, med_pit1, hard_pit1,
                            "2-Stop First Stop — opening compound", config['total_laps']),
                            width='stretch')
        with c3:
            st.plotly_chart(make_compound_hist(soft_pit2, med_pit2, hard_pit2,
                            "2-Stop Second Stop — second stint compound", config['total_laps']),
                            width='stretch')

        # =========================================================================
        # COMPOUND DISTRIBUTION BAR CHARTS
        # Top 6 compound combos by frequency.
        # =========================================================================
        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.subheader("1-Stop Compound Distribution")
            compound_counts_1 = Counter(optimal_compounds_1_stop)
            top_combos = compound_counts_1.most_common(6)
            labels = [f"{c[0][0].upper()}→{c[1][0].upper()}" for c, _ in top_combos]
            values = [v for _, v in top_combos]
            fig = px.bar(x=labels, y=values,
                         labels={'x': 'Compound Combination', 'y': 'Frequency'},
                         title=f"Best 1-Stop Compound Distribution ({n_simulations:,} simulations)")
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.subheader("2-Stop Compound Distribution")
            compound_counts_2 = Counter(optimal_compounds_2_stop)
            top_combos = compound_counts_2.most_common(6)
            labels = [f"{c[0][0].upper()}→{c[1][0].upper()}→{c[2][0].upper()}" for c, _ in top_combos]
            values = [v for _, v in top_combos]
            fig = px.bar(x=labels, y=values,
                         labels={'x': 'Compound Combination', 'y': 'Frequency'},
                         title=f"Best 2-Stop Compound Distribution ({n_simulations:,} simulations)")
            st.plotly_chart(fig, width='stretch')

        # =========================================================================
        # LAP TRACES — Plotly interactive
        # Stint regions shaded by compound, pit stops marked with vlines.
        # Lap trace in royalblue — readable on both light and dark backgrounds.
        # These are also saved to plot_buffers as matplotlib PNGs for download.
        # =========================================================================
        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.subheader("Best 1-Stop Lap Trace")
            fig = plot_single_race_st(config, [best_1stop_pit], best_1stop_compounds)
            st.plotly_chart(fig, width='stretch')

        with c2:
            st.subheader("Best 2-Stop Lap Trace")
            fig = plot_single_race_st(config, [best_2stop_pit1, best_2stop_pit2], best_2stop_compounds)
            st.plotly_chart(fig, width='stretch')

        # Generate matplotlib versions of traces for PNG download
        # (Plotly PNG export requires kaleido which isn't available on Streamlit Cloud)
        compound_colors_mpl = {'soft': 'red', 'medium': 'yellow', 'hard': 'gray'}

        for trace_key, pit_laps, compounds in [
            ('1-Stop Lap Trace', [best_1stop_pit], best_1stop_compounds),
            ('2-Stop Lap Trace', [best_2stop_pit1, best_2stop_pit2], best_2stop_compounds)
        ]:
            sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
            _, lap_times = simulate_race(pit_laps[0], compounds, sc_laps, config,
                                         pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)
            stint_boundaries = [0] + pit_laps + [config['total_laps']]
            fig_mpl, ax = plt.subplots(dpi=plot_dpi)
            ax.plot(range(1, config['total_laps'] + 1), lap_times, color='royalblue')
            for pit in pit_laps:
                ax.axvline(x=pit, color='red', linestyle='--', label=f'Pit lap {pit}')
            for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
                ax.axvspan(start, end, alpha=0.2, color=compound_colors_mpl[compounds[i]],
                           label=f'Stint {i+1}: {compounds[i][0].upper()}')
            ax.set_xlabel('Lap')
            ax.set_ylabel('Lap time (s)')
            ax.set_title('Single race lap time trace')
            ax.set_ylim(config['base_lap_time'] - 2, config['base_lap_time'] + config['pit_loss'] + 10)
            ax.legend()
            buf = io.BytesIO()
            fig_mpl.savefig(buf, format='png', dpi=plot_dpi, bbox_inches='tight')
            buf.seek(0)
            plot_buffers[trace_key] = buf
            plt.close(fig_mpl)

        # =========================================================================
        # STRATEGY TABLES
        # Built from winning_strategies only — win rate is within strategy subset.
        # Tables use same rank 1 strategy as lap traces for consistency.
        # =========================================================================
        st.subheader("Strategy Tables")

        total_1stop_wins = len(winning_strategies_1_stop)
        rows_1 = []
        # table building — strip the time from the tuple
        pit_lap_counts_1 = Counter({(pit, comp): len(v) for (pit, comp), v in times_by_strategy_1.items()})
        for (pit, comp), freq in pit_lap_counts_1.most_common(10):
            rows_1.append({
                'Starting Tire': comp[0][0].upper(),
                'Stop 1':        f"Lap {pit} → {comp[1][0].upper()}",
                'Frequency':     freq,
                'Win Rate':      f"{freq/total_1stop_wins*100:.1f}%" if total_1stop_wins > 0 else "N/A"
            })
        df_1stop = pd.DataFrame(rows_1)
        df_1stop.index = range(1, len(df_1stop) + 1)
        df_1stop.index.name = 'Rank'

        total_2stop_wins = len(winning_strategies_2_stop)
        rows_2 = []
        # table building — strip the time from the tuple
        pit_lap_counts_2 = Counter({k: len(v) for k, v in times_by_strategy_2.items()})
        for (pit1, pit2, comp), freq in pit_lap_counts_2.most_common(10):
            rows_2.append({
                'Starting Tire': comp[0][0].upper(),
                'Stop 1':        f"Lap {pit1} → {comp[1][0].upper()}",
                'Stop 2':        f"Lap {pit2} → {comp[2][0].upper()}",
                'Frequency':     freq,
                'Win Rate':      f"{freq/total_2stop_wins*100:.1f}%" if total_2stop_wins > 0 else "N/A"
            })
        df_2stop = pd.DataFrame(rows_2)
        df_2stop.index = range(1, len(df_2stop) + 1)
        df_2stop.index.name = 'Rank'

        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.write("**Top 10 1-Stop Strategies**")
            st.write(f"*{total_1stop_wins} total 1-stop wins ({total_1stop_wins/n_simulations*100:.1f}% of simulations)*")
            st.caption("Win rate = % of wins within this strategy type, not overall race wins")
            st.dataframe(df_1stop, width='stretch')

        with c2:
            st.write("**Top 10 2-Stop Strategies**")
            st.write(f"*{total_2stop_wins} total 2-stop wins ({total_2stop_wins/n_simulations*100:.1f}% of simulations)*")
            st.caption("Win rate = % of wins within this strategy type, not overall race wins")
            st.dataframe(df_2stop, width='stretch')

        # =========================================================================
        # DOWNLOAD SECTION
        # Plotly charts can be downloaded individually via the camera icon.
        # Lap traces (matplotlib PNG) and CSV tables are bundled in a zip.
        # =========================================================================
        csv_1stop  = df_1stop.to_csv().encode('utf-8')
        csv_2stop  = df_2stop.to_csv().encode('utf-8')

        config_rows = [{'Parameter': k, 'Value': v}
                       for k, v in config.items() if k != 'compound_data']
        for comp, data in config['compound_data'].items():
            for param, val in data.items():
                config_rows.append({'Parameter': f'{comp}_{param}', 'Value': val})
        csv_config = pd.DataFrame(config_rows).to_csv(index=False).encode('utf-8')

        st.subheader("Download Results")
        st.caption("Interactive plots can be downloaded individually using the camera icon on each chart. Lap traces and tables are available below.")

        with st.expander("Select items to include", expanded=False):
            dl_1stop_table = st.checkbox("1-Stop Strategy Table",   value=True)
            dl_2stop_table = st.checkbox("2-Stop Strategy Table",   value=True)
            dl_config      = st.checkbox("Simulation Parameters",   value=True)
            dl_1stop_trace = st.checkbox("1-Stop Lap Trace (PNG)",  value=True)
            dl_2stop_trace = st.checkbox("2-Stop Lap Trace (PNG)",  value=True)

        # Map buffer keys to checkbox variables
        checkbox_map = {
            '1-Stop Lap Trace': dl_1stop_trace,
            '2-Stop Lap Trace': dl_2stop_trace,
        }

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w') as zf:
            if dl_1stop_table:
                zf.writestr("1stop_strategies.csv", csv_1stop)
            if dl_2stop_table:
                zf.writestr("2stop_strategies.csv", csv_2stop)
            if dl_config:
                zf.writestr("parameters.csv", csv_config)
            for name, buf in plot_buffers.items():
                if checkbox_map.get(name, False):
                    zf.writestr(f"{name}.png", buf.getvalue())
        zip_buf.seek(0)

        track_name = track.replace(" ", "_").replace("(", "").replace(")", "")
        st.download_button(
            label="Download Selected Results",
            data=zip_buf,
            file_name=f"f1_simulation_{track_name}.zip",
            mime="application/zip"
        )