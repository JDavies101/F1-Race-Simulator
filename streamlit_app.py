import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from monte_carlo_medium_v1 import simulate_race, run_simulation, generate_safety_cars
from track_presets import TRACK_PRESETS

st.set_page_config(layout="wide", page_title="F1 Strategy Simulator")
st.title("F1 Race Strategy Simulator")
st.write("By Jake Davies")

# Sidebar Configuration
st.sidebar.header("Race Parameters")
mobile_mode = st.sidebar.toggle("Mobile layout", value=False)

track = st.sidebar.selectbox("Track Preset", list(TRACK_PRESETS.keys()))

if track != "Custom" and st.sidebar.button("Load Preset"):
    preset = TRACK_PRESETS[track]
    for key, val in preset.items():
        st.session_state[key] = val
    st.rerun()

base_lap_time = st.sidebar.slider("Base Lap Time (s)", 55, 150, st.session_state.get('base_lap_time', 80), key='base_lap_time')
total_laps = st.sidebar.slider("Total Laps", 40, 90, st.session_state.get('total_laps', 60), key='total_laps')
pit_loss = st.sidebar.slider("Pit Loss (s)", 15, 35, st.session_state.get('pit_loss', 25), key='pit_loss')
sc_chance_pct = st.sidebar.slider("Safety Car Chance (%)", 0, 100, st.session_state.get('sc_chance_pct', 50), key='sc_chance_pct') 
n_simulations = st.sidebar.slider("Simulations", 100, 10000, st.session_state.get('n_simulations', 500), step=100, key='n_simulations')
search_method = st.sidebar.radio("Search Method", ["sampled", "exhaustive"])

with st.sidebar.expander("Compound Settings"):
    st.write("**Soft**")
    soft_offset = st.slider("Offset (s)", -3.0, 0.0, st.session_state.get('soft_offset', -1.0), step=0.1, key="soft_offset")
    soft_deg = st.slider("Deg rate", 0.01, 0.15, st.session_state.get('soft_deg', 0.08), step=0.01, key="soft_deg")
    soft_max = st.slider("Max laps", 5, 30, st.session_state.get('soft_max', 15), key="soft_max")
    
    st.write("**Medium**")
    med_offset = st.slider("Offset (s)", -1.0, 1.0, st.session_state.get('med_offset', 0.0), step=0.1, key="med_offset")
    med_deg = st.slider("Deg rate", 0.01, 0.15, st.session_state.get('med_deg', 0.05), step=0.01, key="med_deg")
    med_max = st.slider("Max laps", 20, 50, st.session_state.get('med_max', 35), key="med_max")
    
    st.write("**Hard**")
    hard_offset = st.slider("Offset (s)", 0.0, 3.0, st.session_state.get('hard_offset', 1.5), step=0.1, key="hard_offset")
    hard_deg = st.slider("Deg rate", 0.01, 0.10, st.session_state.get('hard_deg', 0.02), step=0.01, key="hard_deg")
    hard_max = st.slider("Max laps", 30, 70, st.session_state.get('hard_max', 50), key="hard_max")

with st.sidebar.expander("Traffic & Safety Car Settings"):
    base_traffic_prob = st.slider("Base traffic probability", 0.0, 1.0, st.session_state.get('base_traffic_prob', 0.4), step=0.05, key="base_traffic_prob")
    spread_rate = st.slider("Traffic spread rate", 5, 40, st.session_state.get('spread_rate', 15), key="spread_rate")
    traffic_time_loss = st.slider("Traffic time loss (s)", 0.1, 2.0, st.session_state.get('traffic_time_loss', 0.5), step=0.1, key="traffic_loss")
    sc_pit_loss = st.slider("SC pit loss (s)", 5, 20, st.session_state.get('sc_pit_loss', 12), key="sc_pit_loss")
    sc_base_lap_time = st.slider("SC lap time (s)", 80, 120, st.session_state.get('sc_base_lap_time', 90), key="sc_lap_time")

config = {
    
    'base_lap_time': base_lap_time,
    'pit_loss': pit_loss,
    'sc_pit_loss': sc_pit_loss,
    'sc_base_lap_time': sc_base_lap_time,
    'total_laps': total_laps,
    'sc_chance': sc_chance_pct/total_laps/100,
    'base_traffic_prob': base_traffic_prob,
    'spread_rate': spread_rate,
    'traffic_time_loss': traffic_time_loss,
    'compound_data': {
    'soft':   {'lap_time_offset': soft_offset, 'deg_rate': soft_deg, 'max_laps': soft_max},
    'medium': {'lap_time_offset': med_offset,  'deg_rate': med_deg,  'max_laps': med_max},
    'hard':   {'lap_time_offset': hard_offset, 'deg_rate': hard_deg, 'max_laps': hard_max},
},
    'search_method': search_method,
    'n_strategies_sampled': 500
    
}

plot_dpi = 2000

def plot_single_race_st(config, pit_laps, compounds):
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
    _, lap_times = simulate_race(
        pit_laps[0], compounds, sc_laps, config,
        pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)
    
    compound_colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'gray'}
    stint_boundaries = [0] + pit_laps + [config['total_laps']]
    
    fig, ax = plt.subplots(dpi = plot_dpi)
    ax.plot(range(1, config['total_laps']+1), lap_times)
    for pit in pit_laps:
        ax.axvline(x=pit, color='red', linestyle='--', label=f'Pit lap {pit}')
    for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
        ax.axvspan(start, end, alpha=0.2, color=compound_colors[compounds[i]],
                   label=f'Stint {i+1}: {compounds[i][0].upper()}')
    ax.set_xlabel('Lap')
    ax.set_ylabel('Lap time (s)')
    ax.set_title('Single race lap time trace')
    ax.set_ylim(config['base_lap_time']-2, config['base_lap_time']+config['pit_loss']+10)
    ax.legend()
    return fig

if st.button("Run Simulation"):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    args = [(config, sim) for sim in range(n_simulations)]
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    sim_results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_simulation, arg): i for i, arg in enumerate(args)}
        for i, future in enumerate(as_completed(futures)):
            sim_results.append(future.result())
            progress = (i+1) / n_simulations
            progress_bar.progress(progress)
            status_text.text(f"Simulation {i+1}/{n_simulations}")
            
    st.session_state['results'] = sim_results
    st.session_state['config'] = config
    st.success(f"Done! {n_simulations} simulations complete.")
    
if 'results' in st.session_state:
    sim_results = st.session_state['results']
    config = st.session_state['config']
    
    optimal_laps = []
    optimal_laps_1_stop = []
    optimal_laps_2_stop_pit1 = []
    optimal_laps_2_stop_pit2 = []
    optimal_compounds_1_stop = []
    optimal_compounds_2_stop = []
    best_times_1_stop = []
    best_times_2_stop = []
    
    for best_time_1, best_time_2, best_1, best_2, _, _ in sim_results:
        pit1, c1 = best_1
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
        
    most_common_strategy = max(set(optimal_laps), key=optimal_laps.count)
    
    avg_time_1stop = np.mean(best_times_1_stop)
    avg_time_2stop = np.mean(best_times_2_stop)
    
    if mobile_mode:
        
        st.write("Most Common Strategy", most_common_strategy)
        st.write("1-Stop Win Rate", f"{optimal_laps.count('1-stop')}/{n_simulations}")
        st.write("2-Stop Win Rate", f"{optimal_laps.count('2-stop')}/{n_simulations}")
        st.write("Avg Best 1-Stop Time", f"{avg_time_1stop:.1f}s")
        st.write("Avg Best 2-Stop Time", f"{avg_time_2stop:.1f}s")
        
    else:
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Most Common Strategy", most_common_strategy)
        col2.metric("1-Stop Win Rate", f"{optimal_laps.count('1-stop')}/{n_simulations}")
        col3.metric("2-Stop Win Rate", f"{optimal_laps.count('2-stop')}/{n_simulations}")
        col4.metric("Avg Best 1-Stop Time", f"{avg_time_1stop:.1f}s")
        col5.metric("Avg Best 2-Stop Time", f"{avg_time_2stop:.1f}s")
    
    from collections import Counter
    
    best_1stop_compounds = list(Counter(optimal_compounds_1_stop).most_common(1)[0][0])
    filtered_pit1_1stop = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if list(c) == best_1stop_compounds]
    best_1stop_pit = Counter(filtered_pit1_1stop).most_common(1)[0][0]

    # get most common compound combo
    best_2stop_compounds = list(Counter(optimal_compounds_2_stop).most_common(1)[0][0])

    # filter pit laps to only those simulations that used that compound combo
    filtered_pit1 = [p1 for p1, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if list(c) == best_2stop_compounds]
    filtered_pit2 = [p2 for p2, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if list(c) == best_2stop_compounds]

    best_2stop_pit1 = Counter(filtered_pit1).most_common(1)[0][0]
    best_2stop_pit2 = Counter(filtered_pit2).most_common(1)[0][0]
    
    st.subheader("Strategy Summary")
    
    if mobile_mode:
        
        st.write(f"**Best 1-Stop:** Pit lap {best_1stop_pit}, compounds {' → '.join([c[0].upper() for c in best_1stop_compounds])}")
        st.write(f"**Avg race time:** {avg_time_1stop:.1f}s ({avg_time_1stop//60:.0f}m {avg_time_1stop%60:.1f}s)")
        st.write(f"**Best 2-Stop:** Pit laps {best_2stop_pit1}/{best_2stop_pit2}, compounds {' → '.join([c[0].upper() for c in best_2stop_compounds])}")
        st.write(f"**Avg race time:** {avg_time_2stop:.1f}s ({avg_time_2stop//60:.0f}m {avg_time_2stop%60:.1f}s)")
        
    else:
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Best 1-Stop:** Pit lap {best_1stop_pit}, compounds {' → '.join([c[0].upper() for c in best_1stop_compounds])}")
            st.write(f"**Avg race time:** {avg_time_1stop:.1f}s ({avg_time_1stop//60:.0f}m {avg_time_1stop%60:.1f}s)")
        with col2:
            st.write(f"**Best 2-Stop:** Pit laps {best_2stop_pit1}/{best_2stop_pit2}, compounds {' → '.join([c[0].upper() for c in best_2stop_compounds])}")
            st.write(f"**Avg race time:** {avg_time_2stop:.1f}s ({avg_time_2stop//60:.0f}m {avg_time_2stop%60:.1f}s)")
    
    st.subheader("Strategy Distributions")
    counts = Counter(optimal_laps)
    fig, ax = plt.subplots(dpi = plot_dpi)
    ax.bar(counts.keys(), counts.values())
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Best Pit Strategy Distribution ({n_simulations:,} simulations)")
    st.pyplot(fig)
    plt.close(fig)
    
    if mobile_mode:
        
        st.subheader("1-Stop Pit Lap Distribution")
        fig, ax = plt.subplots(dpi = plot_dpi)
        ax.hist(optimal_laps_1_stop, bins=range(1, config['total_laps']), edgecolor='black')
        ax.axvline(x=config['total_laps']/2, color='red', linestyle='--', label='Deterministic optimum')
        ax.set_xlabel("Optimal Pit Lap")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("2-Stop Pit Lap Distribution")
        fig, ax = plt.subplots(dpi = plot_dpi)
        ax.hist(optimal_laps_2_stop_pit1, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='First stop')
        ax.hist(optimal_laps_2_stop_pit2, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='Second stop')
        ax.axvline(x=config['total_laps']/3, color='red', linestyle='--', label='Deterministic optimum, stop 1')
        ax.axvline(x=2*config['total_laps']/3, color='red', linestyle='--', label='Deterministic optimum, stop 2')
        ax.set_xlabel("Optimal Pit Lap")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
    else:
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1-Stop Pit Lap Distribution")
            fig, ax = plt.subplots(dpi = plot_dpi)
            ax.hist(optimal_laps_1_stop, bins=range(1, config['total_laps']), edgecolor='black')
            ax.axvline(x=config['total_laps']/2, color='red', linestyle='--', label='Deterministic optimum')
            ax.set_xlabel("Optimal Pit Lap")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("2-Stop Pit Lap Distribution")
            fig, ax = plt.subplots(dpi = plot_dpi)
            ax.hist(optimal_laps_2_stop_pit1, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='First stop')
            ax.hist(optimal_laps_2_stop_pit2, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='Second stop')
            ax.axvline(x=config['total_laps']/3, color='red', linestyle='--', label='Deterministic optimum, stop 1')
            ax.axvline(x=2*config['total_laps']/3, color='red', linestyle='--', label='Deterministic optimum, stop 2')
            ax.set_xlabel("Optimal Pit Lap")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    if mobile_mode:
        
        st.subheader("1-Stop Compound Distribution")
        compound_counts_1 = Counter(optimal_compounds_1_stop)
        top_combos = compound_counts_1.most_common(6)
        labels = [f"{c[0][0].upper()}→{c[1][0].upper()}" for c, _ in top_combos]
        values = [v for _, v in top_combos]
        fig, ax = plt.subplots(dpi = plot_dpi)
        ax.bar(labels, values)
        ax.set_xlabel("Compound Combination")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Best 1-Stop Compound Distribution ({n_simulations:,} simulations)")
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("2-Stop Compound Distribution")
        compound_counts_2 = Counter(optimal_compounds_2_stop)
        top_combos = compound_counts_2.most_common(6)
        labels = [f"{c[0][0].upper()}→{c[1][0].upper()}→{c[2][0].upper()}" for c, _ in top_combos]
        values = [v for _, v in top_combos]
        fig, ax = plt.subplots(dpi = plot_dpi)
        ax.bar(labels, values)
        ax.set_xlabel("Compound Combination")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Best 2-Stop Compound Distribution ({n_simulations:,} simulations)")
        st.pyplot(fig)
        plt.close(fig)
        
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1-Stop Compound Distribution")
            compound_counts_1 = Counter(optimal_compounds_1_stop)
            top_combos = compound_counts_1.most_common(6)
            labels = [f"{c[0][0].upper()}→{c[1][0].upper()}" for c, _ in top_combos]
            values = [v for _, v in top_combos]
            fig, ax = plt.subplots(dpi = plot_dpi)
            ax.bar(labels, values)
            ax.set_xlabel("Compound Combination")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Best 1-Stop Compound Distribution ({n_simulations:,} simulations)")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("2-Stop Compound Distribution")
            compound_counts_2 = Counter(optimal_compounds_2_stop)
            top_combos = compound_counts_2.most_common(6)
            labels = [f"{c[0][0].upper()}→{c[1][0].upper()}→{c[2][0].upper()}" for c, _ in top_combos]
            values = [v for _, v in top_combos]
            fig, ax = plt.subplots(dpi = plot_dpi)
            ax.bar(labels, values)
            ax.set_xlabel("Compound Combination")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Best 2-Stop Compound Distribution ({n_simulations:,} simulations)")
            st.pyplot(fig)
            plt.close(fig)
        
    if mobile_mode:
        
        st.subheader("Best 1-Stop Lap Trace")
        fig = plot_single_race_st(config, [best_1stop_pit], best_1stop_compounds)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("Best 2-Stop Lap Trace")
        fig = plot_single_race_st(config, [best_2stop_pit1, best_2stop_pit2], best_2stop_compounds)
        st.pyplot(fig)
        plt.close(fig)
        
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Best 1-Stop Lap Trace")
            fig = plot_single_race_st(config, [best_1stop_pit], best_1stop_compounds)
            st.pyplot(fig)
            plt.close(fig)
            
        with col2:
            st.subheader("Best 2-Stop Lap Trace")
            fig = plot_single_race_st(config, [best_2stop_pit1, best_2stop_pit2], best_2stop_compounds)
            st.pyplot(fig)
            plt.close(fig)