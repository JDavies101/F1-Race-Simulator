# =============================================================================
# F1 RACE STRATEGY SIMULATOR — STREAMLIT WEB APP
# =============================================================================
# Interactive front-end for the Monte Carlo race strategy simulator.
# Users can adjust all race parameters via sidebar sliders, load track presets
# for all 24 rounds of the 2026 F1 calendar, run parallel simulations, and
# view strategy distribution plots and lap time traces in real time.
#
# Architecture:
#   - Sidebar: all config inputs (sliders, radio, expanders, track preset loader)
#   - Run button: triggers parallel simulation via ProcessPoolExecutor
#   - st.session_state: persists results across Streamlit reruns (slider moves)
#   - Results section: metrics, strategy summary, and 5 plot types
#   - Mobile toggle: switches between single-column and two-column layouts
#
# Key Streamlit behavior to understand:
#   Every user interaction (slider move, button click) reruns the ENTIRE script
#   from top to bottom. Session state is used to preserve results between reruns
#   so the user doesn't lose their simulation output when adjusting a slider.
# =============================================================================

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import io
import zipfile
from monte_carlo_medium_v1 import simulate_race, run_simulation, generate_safety_cars
from track_presets import TRACK_PRESETS

# -----------------------------------------------------------------------------
# PAGE CONFIG
# Must be the very first Streamlit call — Streamlit throws an error if anything
# else runs before set_page_config(). 'wide' layout uses full browser width.
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="F1 Strategy Simulator")
st.title("F1 Race Strategy Simulator")
st.write("By Jake Davies")

# =============================================================================
# SIDEBAR: CONFIGURATION INPUTS
# All parameters flow through the config dict into run_simulation().
# Sliders use key= so Streamlit manages their values in session_state,
# which allows the track preset loader to update them programmatically.
# =============================================================================

st.sidebar.header("Race Parameters")

# Mobile layout toggle — switches between 2-column (desktop) and 1-column (mobile)
# placed at the top of sidebar so it's always accessible
mobile_mode = st.sidebar.toggle("Mobile layout", value=False)

# --- Track Preset Loader ---
# Selecting a track and clicking Load Preset writes all preset values into
# session_state, then st.rerun() triggers a full script rerun so all sliders
# update to the preset values on the next render.
track = st.sidebar.selectbox("Track Preset", list(TRACK_PRESETS.keys()))

if track != "Custom" and st.sidebar.button("Load Preset"):
    preset = TRACK_PRESETS[track]
    for key, val in preset.items():
        st.session_state[key] = val  # write preset values into session state
    st.session_state.pop('results', None)  # clear results when preset loads
    st.rerun()                        # rerun so sliders pick up new session state values

# --- Main Race Parameters ---
# st.session_state.get(key, default) reads preset-loaded values if present,
# otherwise falls back to the hardcoded default on first load.
base_lap_time = st.sidebar.slider("Base Lap Time (s)", 55, 150, st.session_state.get('base_lap_time', 80), key='base_lap_time')
total_laps    = st.sidebar.slider("Total Laps", 40, 90, st.session_state.get('total_laps', 60), key='total_laps')
pit_loss      = st.sidebar.slider("Pit Loss (s)", 15, 35, st.session_state.get('pit_loss', 25), key='pit_loss')

# SC chance entered as % per race (0-100), converted to per-lap probability in config
sc_chance_pct = st.sidebar.slider("Safety Car Chance (%)", 0, 100, st.session_state.get('sc_chance_pct', 50), key='sc_chance_pct')

# More simulations = smoother distributions but slower runtime
n_simulations = st.sidebar.slider("Simulations", 100, 10000, st.session_state.get('n_simulations', 500), step=100, key='n_simulations')

# Exhaustive evaluates all pit lap/compound combos (optimal but slow)
# Sampled randomly draws N strategies (fast, approximate — better for interactive use)
search_method = st.sidebar.radio("Search Method", ["sampled", "exhaustive"])

# --- Compound Settings Expander ---
# Collapsed by default to keep sidebar clean. Each slider has a unique key=
# to prevent Streamlit confusing sliders with identical labels across compounds.
with st.sidebar.expander("Compound Settings"):
    st.write("**Soft**")
    soft_offset = st.slider("Offset (s)", -3.0, 0.0, st.session_state.get('soft_offset', -1.0), step=0.1, key="soft_offset")
    # lap_time_offset: speed advantage vs medium baseline (negative = faster)
    soft_deg    = st.slider("Deg rate", 0.01, 0.15, st.session_state.get('soft_deg', 0.08), step=0.01, key="soft_deg")
    # deg_rate: seconds of lap time added per lap of tire age
    soft_max    = st.slider("Max laps", 5, 30, st.session_state.get('soft_max', 15), key="soft_max")
    # max_laps: tire age beyond which deg_rate triples (cliff behavior)

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
    # Probability of being stuck behind a slower car at the start of a stint
    # Decays exponentially as the field spreads out (see spread_rate)

    spread_rate = st.slider("Traffic spread rate", 5, 40,
                            st.session_state.get('spread_rate', 15), key="spread_rate")
    # Exponential decay constant for traffic probability: higher = slower decay
    # traffic_prob = base_traffic_prob * exp(-stint_lap / spread_rate)

    traffic_time_loss = st.slider("Traffic time loss (s)", 0.1, 2.0,
                                  st.session_state.get('traffic_time_loss', 0.5), step=0.1, key="traffic_loss")
    # Seconds added to lap time when stuck in traffic

    sc_pit_loss = st.slider("SC pit loss (s)", 5, 20,
                            st.session_state.get('sc_pit_loss', 12), key="sc_pit_loss")
    # Reduced pit loss under safety car — pitting is cheaper relative to cars staying out

    sc_base_lap_time = st.slider("SC lap time (s)", 80, 120,
                                 st.session_state.get('sc_base_lap_time', 90), key="sc_lap_time")
    # All cars neutralized to this lap time during safety car periods

# Short intro — always visible
st.markdown("""
This simulator uses **Monte Carlo methods** to find the optimal pit stop strategy for an F1 race.
Rather than a single deterministic answer, it runs thousands of randomized race scenarios to produce
a *distribution* of optimal strategies — reflecting the uncertainty real strategists face on race day.
""")

# Full detail — collapsed by default
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
# Assembles all sidebar values into the config dict passed to run_simulation().
# sc_chance converts the per-race percentage to a per-lap probability:
#   sc_chance_pct / total_laps / 100
#   e.g. 50% chance per race over 60 laps = 0.83% per lap
# =============================================================================
config = {
    'base_lap_time':    base_lap_time,
    'pit_loss':         pit_loss,
    'sc_pit_loss':      sc_pit_loss,
    'sc_base_lap_time': sc_base_lap_time,
    'total_laps':       total_laps,
    'sc_chance':        sc_chance_pct / total_laps / 100,  # convert % per race to probability per lap
    'base_traffic_prob': base_traffic_prob,
    'spread_rate':      spread_rate,
    'traffic_time_loss': traffic_time_loss,
    'compound_data': {
        'soft':   {'lap_time_offset': soft_offset, 'deg_rate': soft_deg, 'max_laps': soft_max},
        'medium': {'lap_time_offset': med_offset,  'deg_rate': med_deg,  'max_laps': med_max},
        'hard':   {'lap_time_offset': hard_offset, 'deg_rate': hard_deg, 'max_laps': hard_max},
    },
    'search_method':        search_method,
    'n_strategies_sampled': 500,  # used only in sampled mode
}

# DPI for all matplotlib figures rendered in Streamlit
# High DPI = crisp plots. width='stretch' handles actual display sizing.
plot_dpi = 500

# =============================================================================
# LAP TRACE PLOT FUNCTION (Streamlit version)
# =============================================================================

def plot_single_race_st(config, pit_laps, compounds):
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
    _, lap_times = simulate_race(
        pit_laps[0], compounds, sc_laps, config,
        pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)

    compound_colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'gray'}
    stint_boundaries = [0] + pit_laps + [config['total_laps']]

    fig = go.Figure()

    # shade stints
    for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
        color = compound_colors[compounds[i]]
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.15,
                      layer='below', line_width=0)
        # invisible scatter for legend entry
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=color, size=10),
                                 name=f'Stint {i+1}: {compounds[i][0].upper()}'))

    # pit stop lines
    for pit in pit_laps:
        fig.add_vline(x=pit, line_dash='dash', line_color='red',
                      annotation_text=f'Pit lap {pit}')

    # lap time trace
    fig.add_trace(go.Scatter(
        x=list(range(1, config['total_laps']+1)),
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
        yaxis_range=[config['base_lap_time']-2,
                     config['base_lap_time']+config['pit_loss']+10]
    )
    return fig

# =============================================================================
# RUN SIMULATION BUTTON
# =============================================================================
# ProcessPoolExecutor is used instead of multiprocessing.Pool because it
# integrates more cleanly with Streamlit's threading model.
#
# as_completed() yields futures as they finish (not in submission order),
# which allows real-time progress bar updates — pool.map() would block until
# all simulations complete before returning anything.
#
# Results and config are stored in session_state so they persist across
# reruns triggered by slider interactions after simulation completes.
# =============================================================================

if st.button("Run Simulation"):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = [(config, sim) for sim in range(n_simulations)]
    progress_bar = st.progress(0)       # progress bar widget (0.0 to 1.0)
    status_text  = st.empty()           # placeholder text updated each iteration

    sim_results = []
    with ProcessPoolExecutor() as executor:
        # Submit all simulations as independent futures
        futures = {executor.submit(run_simulation, arg): i for i, arg in enumerate(args)}

        # Process results as they complete, updating progress bar in real time
        for i, future in enumerate(as_completed(futures)):
            sim_results.append(future.result())
            progress_bar.progress((i + 1) / n_simulations)
            status_text.text(f"Simulation {i+1}/{n_simulations}")

    # Store in session_state — persists across reruns until new simulation is run
    st.session_state['results'] = sim_results
    st.session_state['plot_buffers'] = {}
    st.session_state['config']  = config  # store the config used, not current sidebar state
    st.success(f"Done! {n_simulations} simulations complete.")

# =============================================================================
# RESULTS SECTION
# Renders only when simulation results exist in session_state.
# Uses the stored config (not current sidebar values) so plots always match
# the parameters that were actually used in the simulation.
# =============================================================================

if 'results' in st.session_state:
    with st.spinner("Preparing Results..."):
        sim_results = st.session_state['results']
        plot_buffers = {}
        config      = st.session_state['config']   # use config from when simulation ran

        # --- Unpack results from all simulations ---
        optimal_laps             = []   # winning strategy type per simulation ('1-stop' or '2-stop')
        optimal_laps_1_stop      = []   # best 1-stop pit lap per simulation
        optimal_laps_2_stop_pit1 = []   # best 2-stop first pit lap per simulation
        optimal_laps_2_stop_pit2 = []   # best 2-stop second pit lap per simulation
        optimal_compounds_1_stop = []   # best 1-stop compound combo per simulation
        optimal_compounds_2_stop = []   # best 2-stop compound combo per simulation
        best_times_1_stop        = []   # best 1-stop total race time per simulation
        best_times_2_stop        = []   # best 2-stop total race time per simulation
        winning_strategies_1_stop = []
        winning_strategies_2_stop = []
        plot_buffers = {}  

        for best_time_1, best_time_2, best_1, best_2, _, _ in sim_results:
            pit1, c1       = best_1   # e.g. pit1=20, c1=('soft','medium')
            pit2a, pit2b, c2 = best_2  # e.g. pit2a=15, pit2b=38, c2=('soft','medium','hard')

            # Determine which strategy won this simulation by comparing race times
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
            
            # only record if this strategy type actually won overall
            if best[1] == '1-stop':
                winning_strategies_1_stop.append((pit1, c1))
            else:
                winning_strategies_2_stop.append((pit2a, pit2b, c2))

        # Overall most common winning strategy across all simulations
        most_common_strategy = max(set(optimal_laps), key=optimal_laps.count)

        # Average best race times across simulations
        avg_time_1stop = np.mean(best_times_1_stop)
        avg_time_2stop = np.mean(best_times_2_stop)

        # --- Find representative strategies for lap traces ---
        # Step 1: Find most common compound combo (aggregated across all pit laps)
        # Step 2: Filter pit laps to only simulations that used that combo
        # Step 3: Find most common pit lap within that filtered subset
        # This two-step approach ensures the lap trace compound and pit lap are consistent,
        # rather than finding them independently (which can give mismatched results).
        # --- Find representative strategies for lap traces ---
        # Pulled directly from rank 1 of the winning strategy counters
        # so the lap trace always matches the strategy table exactly.
        from collections import Counter

        pit_lap_counts_1 = Counter(winning_strategies_1_stop)
        pit_lap_counts_2 = Counter(winning_strategies_2_stop)

        # fallback if no 1-stop wins
        if pit_lap_counts_1:
            top_1stop            = pit_lap_counts_1.most_common(1)[0][0]
            best_1stop_pit       = top_1stop[0]
            best_1stop_compounds = list(top_1stop[1])
        else:
            # fallback if no 1-stop wins — use midpoint and most common compounds
            best_1stop_pit       = config['total_laps'] // 2
            best_1stop_compounds = list(Counter(optimal_compounds_1_stop).most_common(1)[0][0])

        # fallback if no 2-stop wins
        if pit_lap_counts_2:
            top_2stop            = pit_lap_counts_2.most_common(1)[0][0]
            best_2stop_pit1      = top_2stop[0]
            best_2stop_pit2      = top_2stop[1]
            best_2stop_compounds = list(top_2stop[2])
        else:
            best_2stop_pit1      = config['total_laps'] // 3
            best_2stop_pit2      = 2 * config['total_laps'] // 3
            best_2stop_compounds = list(Counter(optimal_compounds_2_stop).most_common(1)[0][0])

        # =========================================================================
        # METRICS ROW
        # st.metric() renders large-font KPI cards. Mobile stacks them vertically.
        # =========================================================================
        if mobile_mode:
            st.write("Most Common Strategy", most_common_strategy)
            st.write("1-Stop Win Rate", f"{optimal_laps.count('1-stop')}/{n_simulations}")
            st.write("2-Stop Win Rate", f"{optimal_laps.count('2-stop')}/{n_simulations}")
            st.write("Avg Best 1-Stop Time", f"{avg_time_1stop:.1f}s")
            st.write("Avg Best 2-Stop Time", f"{avg_time_2stop:.1f}s")
        else:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Most Common Strategy", most_common_strategy)
            col2.metric("1-Stop Win Rate",  f"{optimal_laps.count('1-stop')}/{n_simulations}")
            col3.metric("2-Stop Win Rate",  f"{optimal_laps.count('2-stop')}/{n_simulations}")
            col4.metric("Avg Best 1-Stop Time", f"{avg_time_1stop:.1f}s")
            col5.metric("Avg Best 2-Stop Time", f"{avg_time_2stop:.1f}s")

        # =========================================================================
        # STRATEGY SUMMARY
        # Text summary of the most representative 1-stop and 2-stop strategies.
        # Compound names are abbreviated to first letter (S/M/H) for readability.
        # Mobile mode stacks vertically; desktop mode uses 2 columns.
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
        # STRATEGY DISTRIBUTION PLOT
        # Centered on desktop via [1,2,1] columns; full width on mobile.
        # =========================================================================
        if mobile_mode:
            center = st.container()
        else:
            _, center, _ = st.columns([1, 2, 1])
        with center:
            st.subheader("Strategy Distribution")
            counts = Counter(optimal_laps)
            fig = px.bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                labels={'x': 'Strategy', 'y': 'Frequency'},
                title=f"Best Pit Strategy Distribution ({n_simulations:,} simulations)"
            )
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['Strategy Distribution'] = io.BytesIO(fig.to_image(format='png'))

        # =========================================================================
        # PIT LAP DISTRIBUTION PLOTS
        # Histograms showing which pit lap was optimal across simulations.
        # Vertical lines show the deterministic optimum (equal split across stints).
        # =========================================================================
        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.subheader("1-Stop Pit Lap Distribution")
            fig = px.histogram(
                x=optimal_laps_1_stop,
                nbins=config['total_laps'],
                labels={'x': 'Optimal Pit Lap', 'y': 'Frequency'},
                title=f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)"
            )
            fig.add_vline(x=config['total_laps']/2, line_dash='dash', line_color='red',
                          annotation_text='Deterministic optimum')
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['1-Stop Pit Lap Distribution'] = io.BytesIO(fig.to_image(format='png'))

        with c2:
            st.subheader("2-Stop Pit Lap Distribution")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=optimal_laps_2_stop_pit1, name='First stop', opacity=0.7,
                                       nbinsx=config['total_laps']))
            fig.add_trace(go.Histogram(x=optimal_laps_2_stop_pit2, name='Second stop', opacity=0.7,
                                       nbinsx=config['total_laps']))
            fig.update_layout(barmode='overlay',
                              title=f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)",
                              xaxis_title='Optimal Pit Lap', yaxis_title='Frequency')
            fig.add_vline(x=config['total_laps']/3,   line_dash='dash', line_color='red',
                          annotation_text='Stop 1 optimum')
            fig.add_vline(x=2*config['total_laps']/3, line_dash='dash', line_color='orange',
                          annotation_text='Stop 2 optimum')
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['2-Stop Pit Lap Distribution'] = io.BytesIO(fig.to_image(format='png'))
            
        # =========================================================================
        # PIT LAP BY COMPOUND PLOTS
        # Stacked histograms explaining bimodal peaks by compound choice.
        # =========================================================================
        # Separate pit laps by compound used at each stop
        soft_1stop = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'soft']
        med_1stop  = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'medium']
        hard_1stop = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'hard']
        soft_pit1  = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'soft']
        med_pit1   = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'medium']
        hard_pit1  = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'hard']
        soft_pit2  = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'soft']
        med_pit2   = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'medium']
        hard_pit2  = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'hard']

        def make_compound_hist(soft, med, hard, title):
            """Build a stacked Plotly histogram colored by compound."""
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=soft, name='S', marker_color='red',   opacity=0.8, nbinsx=config['total_laps']))
            fig.add_trace(go.Histogram(x=med,  name='M', marker_color='gold',  opacity=0.8, nbinsx=config['total_laps']))
            fig.add_trace(go.Histogram(x=hard, name='H', marker_color='gray',  opacity=0.8, nbinsx=config['total_laps']))
            fig.update_layout(barmode='stack', title=title,
                              xaxis_title='Optimal Pit Lap', yaxis_title='Frequency')
            return fig

        st.subheader("Pit Lap Distribution by Compound")
        if mobile_mode:
            c1, c2, c3 = st.container(), st.container(), st.container()
        else:
            c1, c2, c3 = st.columns(3)

        with c1:
            fig = make_compound_hist(soft_1stop, med_1stop, hard_1stop,
                            "1-Stop — opening compound")
            st.plotly_chart(fig, width="stretch")
            plot_buffers['1-Stop Stop by Compound'] = io.BytesIO(fig.to_image(format='png'))
        with c2:
            fig = make_compound_hist(soft_pit1, med_pit1, hard_pit1,
                            "2-Stop First Stop — opening compound")
            st.plotly_chart(fig, width="stretch")
            plot_buffers['2-Stop First Stop by Compound'] = io.BytesIO(fig.to_image(format='png')) 
        with c3:
            fig = make_compound_hist(soft_pit2, med_pit2, hard_pit2,
                            "2-Stop Second Stop — second stint compound")
            st.plotly_chart(fig, width="stretch")
            plot_buffers['2-Stop Second Stop by Compound'] = io.BytesIO(fig.to_image(format='png'))

        # =========================================================================
        # COMPOUND DISTRIBUTION PLOTS
        # Bar charts showing which compound combinations won most often.
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
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['1-Stop Compound Bar Chart'] = io.BytesIO(fig.to_image(format='png'))

        with c2:
            st.subheader("2-Stop Compound Distribution")
            compound_counts_2 = Counter(optimal_compounds_2_stop)
            top_combos = compound_counts_2.most_common(6)
            labels = [f"{c[0][0].upper()}→{c[1][0].upper()}→{c[2][0].upper()}" for c, _ in top_combos]
            values = [v for _, v in top_combos]
            fig = px.bar(x=labels, y=values,
                         labels={'x': 'Compound Combination', 'y': 'Frequency'},
                         title=f"Best 2-Stop Compound Distribution ({n_simulations:,} simulations)")
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['2-Stop Compound Bar Chart'] = io.BytesIO(fig.to_image(format='png'))

        # =========================================================================
        # LAP TRACES 
        # Representative race for rank 1 strategy in each table.
        # =========================================================================
        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.subheader("Best 1-Stop Lap Trace")
            fig = plot_single_race_st(config, [best_1stop_pit], best_1stop_compounds)
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['1-Stop Lap Trace'] = io.BytesIO(fig.to_image(format='png'))

        with c2:
            st.subheader("Best 2-Stop Lap Trace")
            fig = plot_single_race_st(config, [best_2stop_pit1, best_2stop_pit2], best_2stop_compounds)
            st.plotly_chart(fig, use_container_width=True)
            plot_buffers['2-Stop Lap Trace'] = io.BytesIO(fig.to_image(format='png'))

        st.subheader("Strategy Tables")

        # --- 1-stop table ---
        total_1stop_wins = len(winning_strategies_1_stop)
        rows_1 = []
        for (pit, comp), freq in pit_lap_counts_1.most_common(10):
            rows_1.append({
                'Starting Tire': comp[0][0].upper(),
                'Stop 1': f"Lap {pit} → {comp[1][0].upper()}",
                'Frequency': freq,
                'Win Rate': f"{freq/total_1stop_wins*100:.1f}%" if total_1stop_wins > 0 else "N/A"
            })
        df_1stop = pd.DataFrame(rows_1)
        df_1stop.index = range(1, len(df_1stop)+1)
        df_1stop.index.name = 'Rank'

        # --- 2-stop table ---
        total_2stop_wins = len(winning_strategies_2_stop)
        rows_2 = []
        for (pit1, pit2, comp), freq in pit_lap_counts_2.most_common(10):
            rows_2.append({
                'Starting Tire': comp[0][0].upper(),
                'Stop 1': f"Lap {pit1} → {comp[1][0].upper()}",
                'Stop 2': f"Lap {pit2} → {comp[2][0].upper()}",
                'Frequency': freq,
                'Win Rate': f"{freq/total_2stop_wins*100:.1f}%" if total_2stop_wins > 0 else "N/A"
            })
        df_2stop = pd.DataFrame(rows_2)
        df_2stop.index = range(1, len(df_2stop)+1)
        df_2stop.index.name = 'Rank'

        if mobile_mode:
            c1, c2 = st.container(), st.container()
        else:
            c1, c2 = st.columns(2)

        with c1:
            st.write("**Top 10 1-Stop Strategies**")
            st.write(f"*{total_1stop_wins} total 1-stop wins ({total_1stop_wins/n_simulations*100:.1f}% of simulations)*")
            st.caption("Win rate = % of wins within this strategy type, not overall race wins")
            st.dataframe(df_1stop, use_container_width=True)

        with c2:
            st.write("**Top 10 2-Stop Strategies**")
            st.write(f"*{total_2stop_wins} total 2-stop wins ({total_2stop_wins/n_simulations*100:.1f}% of simulations)*")
            st.caption("Win rate = % of wins within this strategy type, not overall race wins")
            st.dataframe(df_2stop, use_container_width=True)
                
        csv_1stop = df_1stop.to_csv().encode('utf-8')
        csv_2stop = df_2stop.to_csv().encode('utf-8')
        
        config_rows = [{'Parameter': k, 'Value': v} 
                    for k, v in config.items() 
                    if k != 'compound_data']
        # add compound data rows separately
        for comp, data in config['compound_data'].items():
            for param, val in data.items():
                config_rows.append({'Parameter': f'{comp}_{param}', 'Value': val})

        df_config = pd.DataFrame(config_rows)
        csv_config = df_config.to_csv(index=False).encode('utf-8')
        
        st.subheader("Download Results")

        with st.expander("Select items to include", expanded=False):
            dl_1stop_table   = st.checkbox("1-Stop Strategy Table", value=True)
            dl_2stop_table   = st.checkbox("2-Stop Strategy Table", value=True)
            dl_config        = st.checkbox("Simulation Parameters", value=True)
            dl_strat_dist    = st.checkbox("Strategy Distribution Plot", value=True)
            dl_1stop_pit     = st.checkbox("1-Stop Pit Lap Distribution", value=True)
            dl_2stop_pit     = st.checkbox("2-Stop Pit Lap Distribution", value=True)
            dl_1stop_comp_hist = st.checkbox("1-Stop Stop by Compound", value=True)
            dl_2stop_comp1_hist = st.checkbox("2-Stop First Stop by Compound", value=True)
            dl_2stop_comp2_hist = st.checkbox("2-Stop Second Stop by Compound", value=True)
            dl_1stop_comp_bar  = st.checkbox("1-Stop Compound Bar Chart", value=True)
            dl_2stop_comp_bar  = st.checkbox("2-Stop Compound Bar Chart", value=True)
            dl_1stop_trace   = st.checkbox("1-Stop Lap Trace", value=True)
            dl_2stop_trace   = st.checkbox("2-Stop Lap Trace", value=True)

        checkbox_map = {
            'Strategy Distribution':          dl_strat_dist,
            '1-Stop Pit Lap Distribution':    dl_1stop_pit,
            '2-Stop Pit Lap Distribution':    dl_2stop_pit,
            '1-Stop Stop by Compound':        dl_1stop_comp_hist,
            '2-Stop First Stop by Compound':  dl_2stop_comp1_hist,
            '2-Stop Second Stop by Compound': dl_2stop_comp2_hist,
            '1-Stop Compound Bar Chart':      dl_1stop_comp_bar,
            '2-Stop Compound Bar Chart':      dl_2stop_comp_bar,
            '1-Stop Lap Trace':               dl_1stop_trace,
            '2-Stop Lap Trace':               dl_2stop_trace,
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