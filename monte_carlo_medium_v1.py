# =============================================================================
# F1 MONTE CARLO RACE STRATEGY SIMULATOR
# =============================================================================
# This simulator models a single F1 car's race strategy using Monte Carlo
# methods. Instead of finding one "correct" answer, it runs thousands of
# simulations with randomized events (safety cars, pit stop variability,
# tire deg variance, traffic) to produce a *distribution* of optimal strategies.
#
# Key concepts:
#   - Deterministic baseline: the mathematically optimal pit lap ignoring randomness
#   - Monte Carlo layer: randomness shifts the optimal lap each simulation
#   - Strategy search: exhaustive (all combinations) or sampled (random subset)
#
# Features:
#   - 1-stop and 2-stop strategy comparison
#   - 3 tire compounds (soft, medium, hard) with different deg rates and cliffs
#   - Safety car events reducing effective pit loss
#   - Traffic model with exponential decay (field spreads over time)
#   - Pit stop time variability
#   - Multiprocessing for parallel simulation
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool   # parallel simulation across CPU cores
import sys
from itertools import product      # generates all compound combinations
import random


# =============================================================================
# CORE SIMULATION FUNCTION
# =============================================================================

def simulate_race(pit_lap1, compounds, sc_laps, config, pit_lap2=None):
    """
    Simulate a single race for one car given a fixed strategy.

    This function runs lap-by-lap through the race, computing lap times
    based on tire degradation, compound choice, traffic, and safety car events.
    Randomness is injected per-lap (traffic, deg variance) and per-pit-stop
    (pit loss variability).

    Parameters
    ----------
    pit_lap1 : int
        Lap number of the first (mandatory) pit stop.
    compounds : tuple of str
        Sequence of tire compounds used in each stint.
        e.g. ('soft', 'medium') for 1-stop, ('soft', 'medium', 'hard') for 2-stop.
    sc_laps : np.ndarray of bool
        Boolean array of length total_laps. True = safety car on that lap.
        Generated once per simulation and shared across all strategy evaluations
        so strategies are compared on a fair like-for-like basis.
    config : dict
        All race parameters (see main() for full config definition).
    pit_lap2 : int or None
        Lap number of the optional second pit stop. None = 1-stop strategy.

    Returns
    -------
    total_time : float
        Total race time in seconds.
    lap_times : list of float
        Individual lap times for each of the total_laps laps.
        Used for lap trace visualization.
    """

    total_time = 0
    tire_age = 0       # laps completed on current tire set (resets at each pit)
    stint_lap = 0      # laps into current stint (drives traffic decay, resets at pit)
    stint_number = 0   # which stint we're in (0-indexed, used to look up compound)
    lap_times = []

    # Build list of pit laps, filtering out None for 1-stop races
    pit_laps = [p for p in [pit_lap1, pit_lap2] if p is not None]

    for laps in range(1, config['total_laps'] + 1):

        # --- Randomness draws for this lap ---

        # Pit stop execution variance: some stops are faster/slower (~±2s std)
        # Only applied on pit laps but drawn every lap for simplicity
        pit_loss_randomness = np.random.normal(0, 2)

        # Tire degradation variance: small lap-to-lap noise in deg rate
        tire_degradation_rate_randomness = np.random.normal(0, 0.001)

        # --- Traffic model ---
        # Probability of being stuck behind a slower car decays exponentially
        # as the field spreads out over the stint. High early, low later.
        # After a pit stop, stint_lap resets so traffic probability spikes again
        # (fresh tires = back in traffic as you rejoin the field).
        traffic_prob = config['base_traffic_prob'] * np.exp(-stint_lap / config['spread_rate'])
        in_traffic = np.random.random() < traffic_prob

        # --- Compound lookup ---
        # Determine which stint we're in based on how many pit stops have passed.
        # sum(1 for p in pit_laps if laps > p) counts completed pit stops.
        # e.g. on lap 25 with pit_laps=[20, 40]: 1 stop passed -> stint 1 -> compounds[1]
        stint_number = sum(1 for p in pit_laps if laps > p)
        current_compound = compounds[stint_number]
        compound_data = config['compound_data'][current_compound]

        # --- Tire cliff model ---
        # When tire age exceeds the compound's max_laps, degradation triples.
        # This models real tire "falling off a cliff" behavior where grip drops sharply.
        # e.g. soft tires past lap 15 degrade 3x faster.
        if tire_age > compound_data['max_laps']:
            deg_this_lap = compound_data['deg_rate'] * 3  # cliff multiplier
        else:
            deg_this_lap = compound_data['deg_rate']

        # --- Base lap time ---
        # Under safety car, all cars slow to ~90s laps regardless of tire state.
        # Under green flag, use the configured base lap time.
        if sc_laps[laps - 1]:
            current_base = config['sc_base_lap_time']
        else:
            current_base = config['base_lap_time']

        # --- Lap time calculation ---
        # lap_time = base + compound_offset + degradation_contribution
        # compound_offset: softs are faster (-1s), hards are slower (+1.5s) vs medium baseline
        # degradation: increases linearly with tire age (+ cliff multiplier if over max_laps)
        lap_time = (current_base
                    + compound_data['lap_time_offset']
                    + (deg_this_lap + tire_degradation_rate_randomness) * tire_age)

        # Add traffic penalty if stuck behind a slower car this lap
        # Traffic doesn't apply under SC since all cars are at the same slow pace
        if in_traffic and not sc_laps[laps - 1]:
            lap_time += config['traffic_time_loss']

        # --- Pit stop logic ---
        if laps in pit_laps:
            if sc_laps[laps - 1]:
                # Under safety car: reduced pit loss because all cars are slow
                # The relative time lost vs cars staying out is much smaller
                lap_time += config['sc_pit_loss'] + pit_loss_randomness
            else:
                # Normal green flag pit stop: full pit loss time
                lap_time += config['pit_loss'] + pit_loss_randomness

            # Reset stint counters for next stint
            stint_lap = 0    # traffic decay resets after pit (fresh start in clean air)
            tire_age = 1     # new tire set starts at age 1 (lap 1 of new stint)
        else:
            # No pit stop this lap: increment both age counters
            tire_age += 1
            stint_lap += 1

        lap_times.append(lap_time)

    total_time = sum(lap_times)
    return total_time, lap_times


# =============================================================================
# SIMULATION WORKER FUNCTION (called in parallel by multiprocessing)
# =============================================================================

def run_simulation(args):
    """
    Run one complete simulation: generate a random race scenario, then
    evaluate all candidate strategies against that scenario to find the optimal.

    This function is the unit of parallelism — each worker process runs one
    simulation independently. The key insight is that sc_laps is generated
    ONCE per simulation and shared across all strategy evaluations, ensuring
    fair comparison (all strategies face the same random events).

    Parameters
    ----------
    args : tuple of (config dict, int seed)
        config: race parameters
        seed: random seed for reproducibility (each simulation gets a unique seed)

    Returns
    -------
    Tuple of:
        best_time_1 : float - total race time of best 1-stop strategy
        best_time_2 : float - total race time of best 2-stop strategy
        best_1 : tuple - key of best 1-stop strategy (pit_lap, compounds)
        best_2 : tuple - key of best 2-stop strategy (pit_lap1, pit_lap2, compounds)
        lap_times_1_stop : dict - lap time arrays for all evaluated 1-stop strategies
        lap_times_2_stop : dict - lap time arrays for all evaluated 2-stop strategies
    """

    config, seed = args
    np.random.seed(seed)  # ensures each simulation is reproducible but unique

    # Generate safety car events for this simulation
    # This array is shared across ALL strategy evaluations in this simulation
    # so every strategy faces the same SC scenario — apples-to-apples comparison
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])

    results_1_stop = {}      # key: (pit_lap, compounds_tuple) -> total_time
    results_2_stop = {}      # key: (pit_lap1, pit_lap2, compounds_tuple) -> total_time
    lap_times_1_stop = {}    # key: same as above -> lap_times list
    lap_times_2_stop = {}    # key: same as above -> lap_times list

    # --- Generate valid compound combinations ---
    compound_names = list(config['compound_data'].keys())  # ['soft', 'medium', 'hard']

    # 1-stop: 2 stints, must use at least 2 different compounds -> c1 != c2
    # Results in 6 valid combos: soft/med, soft/hard, med/soft, med/hard, hard/soft, hard/med
    valid_1stop = [(c1, c2) for c1, c2 in product(compound_names, repeat=2) if c1 != c2]

    # 2-stop: 3 stints, must use at least 2 different compounds total
    # Allows soft/soft/medium, soft/medium/soft, etc. but not soft/soft/soft
    valid_2stop = [(c1, c2, c3) for c1, c2, c3 in product(compound_names, repeat=3)
                   if len(set([c1, c2, c3])) >= 2]

    # --- Strategy search ---
    if config['search_method'] == 'exhaustive':
        # Evaluate EVERY valid combination of pit laps and compounds
        # Guarantees finding the true optimum but is computationally expensive
        # ~55 pit laps × 6 compound combos = 330 calls for 1-stop
        # ~1600 pit lap pairs × 15 compound combos = ~24000 calls for 2-stop
        for pit_lap in range(1, config['total_laps'] - 5):
            for c1stop in valid_1stop:
                results_1_stop[(pit_lap, c1stop)], lap_times_1_stop[(pit_lap, c1stop)] = simulate_race(
                    pit_lap, c1stop, sc_laps, config)

            for pit_lap2 in range(pit_lap + 1, config['total_laps'] - 2):
                # pit_lap2 > pit_lap enforced to prevent invalid reverse strategies
                for c2stop in valid_2stop:
                    results_2_stop[(pit_lap, pit_lap2, c2stop)], _ = simulate_race(
                        pit_lap, c2stop, sc_laps, config, pit_lap2=pit_lap2)

    else:
        # Sampled search: randomly draw N strategies instead of evaluating all
        # Much faster, trades exhaustive optimality for speed
        # Useful for interactive Streamlit use where response time matters
        for _ in range(config['n_strategies_sampled']):
            pit_lap1 = np.random.randint(5, config['total_laps'] - 15)
            pit_lap2 = np.random.randint(pit_lap1 + 5, config['total_laps'] - 5)
            n_stops = np.random.choice([1, 2])  # randomly decide 1 or 2 stop
            if n_stops == 1:
                compounds = random.choice(valid_1stop)
                results_1_stop[(pit_lap1, compounds)], _ = simulate_race(
                    pit_lap1, compounds, sc_laps, config)
            else:
                compounds = random.choice(valid_2stop)
                results_2_stop[(pit_lap1, pit_lap2, compounds)], _ = simulate_race(
                    pit_lap1, compounds, sc_laps, config, pit_lap2=pit_lap2)

    # Find the best strategy in each category
    # Guard against empty dicts (possible in sampled mode if all draws were same stop count)
    best_1 = min(results_1_stop, key=results_1_stop.get) if results_1_stop else None
    best_2 = min(results_2_stop, key=results_2_stop.get) if results_2_stop else None

    return results_1_stop[best_1], results_2_stop[best_2], best_1, best_2, lap_times_1_stop, lap_times_2_stop


# =============================================================================
# VISUALIZATION: SINGLE RACE LAP TIME TRACE
# =============================================================================

def plot_single_race(config, pit_laps, compounds, filename='lap_trace.png'):
    """
    Plot lap time trace for a single representative race.

    Shows the sawtooth pattern of tire degradation within each stint,
    pit stop spikes, traffic events, and SC laps. Stint backgrounds are
    color-coded by tire compound.

    Parameters
    ----------
    config : dict
        Race configuration parameters.
    pit_laps : list of int
        Pit stop lap numbers for the strategy to visualize.
    compounds : list of str
        Compound sequence for each stint.
    """

    # Generate a fresh random race scenario for visualization
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])

    total_time, lap_times = simulate_race(
        pit_laps[0], compounds, sc_laps, config,
        pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)

    # Color coding matches real F1 broadcast conventions
    compound_colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'white'}

    # Stint boundaries: [0, pit1, pit2, total_laps] for 2-stop
    stint_boundaries = [0] + pit_laps + [config['total_laps']]

    plt.plot(range(1, config['total_laps'] + 1), lap_times)

    # Mark pit stop laps with vertical lines
    for pit in pit_laps:
        plt.axvline(x=pit, color='red', linestyle='--', label=f'Pit lap {pit}')

    # Shade each stint by compound color
    for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
        plt.axvspan(start, end, alpha=0.2, color=compound_colors[compounds[i]],
                    label=f'Stint {i + 1}: {compounds[i]}')

    plt.xlabel('Lap')
    plt.ylabel('Lap time (s)')
    plt.title('Single race lap time trace')
    plt.legend()
    # Y-axis: base lap time at bottom, pit stop spike at top
    plt.ylim(config['base_lap_time'] - 2, config['base_lap_time'] + config['pit_loss'] + 10)
    plt.savefig(filename, dpi=450, bbox_inches='tight')
    plt.close()


# =============================================================================
# SAFETY CAR GENERATOR
# =============================================================================

def generate_safety_cars(total_laps, sc_chance):
    """
    Generate a boolean array indicating which laps have a safety car.

    Each lap independently has sc_chance probability of SC deployment.
    e.g. sc_chance=0.008 gives ~0.8% per lap ≈ 50% chance of at least
    one SC over a 60-lap race.

    Parameters
    ----------
    total_laps : int
    sc_chance : float - probability per lap (0 to 1)

    Returns
    -------
    np.ndarray of bool, shape (total_laps,)
    """
    return np.random.random(total_laps) < sc_chance


# =============================================================================
# MAIN: CONFIGURATION, SIMULATION LOOP, AND PLOTTING
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # RACE CONFIGURATION
    # All tunable parameters live here. Passing config as a dict allows
    # easy serialization for multiprocessing and future Streamlit integration.
    # -------------------------------------------------------------------------
    config = {
        # Base lap time on medium tires with no deg (seconds)
        'base_lap_time': 81,

        # Time lost in pit lane under green flag conditions (seconds)
        # Represents relative loss to cars staying out at race pace
        'pit_loss': 25,

        # Time lost in pit lane under safety car
        # Lower because all cars are slow, reducing the relative cost of pitting
        'sc_pit_loss': 12,

        # Lap time under safety car (all cars neutralized to this pace)
        'sc_base_lap_time': 90,

        # Race distance in laps
        'total_laps': 60,

        # Probability of safety car deployment per lap
        # 50/60/100 ≈ 0.0083 gives ~50% chance per race across 60 laps
        'sc_chance': 50 / 60 / 100,

        # Traffic model parameters
        'base_traffic_prob': 0.4,    # probability of being in traffic on lap 1 of a stint
        'spread_rate': 15,           # exponential decay constant (higher = slower decay)
        'traffic_time_loss': 0.5,    # seconds lost per lap in traffic

        # Tire compound definitions
        # lap_time_offset: speed vs medium baseline (negative = faster)
        # deg_rate: seconds lost per lap of tire age under normal conditions
        # max_laps: tire age beyond which deg_rate triples (cliff behavior)
        'compound_data': {
            'soft':   {'lap_time_offset': -1.0, 'deg_rate': 0.08, 'max_laps': 15},
            'medium': {'lap_time_offset':  0.0, 'deg_rate': 0.05, 'max_laps': 35},
            'hard':   {'lap_time_offset':  1.5, 'deg_rate': 0.02, 'max_laps': 50},
        },

        # Search method: 'exhaustive' evaluates all combinations (slow, optimal)
        # 'sampled' randomly samples N strategies (fast, approximate)
        'search_method': 'sampled',

        # Number of strategies to sample per simulation (sampled mode only)
        'n_strategies_sampled': 500,
    }

    # Number of Monte Carlo simulations to run
    # More simulations = smoother distributions but longer runtime
    n_simulations = 1000

    # Collect results across all simulations
    optimal_laps = []              # winning strategy type ('1-stop' or '2-stop')
    optimal_laps_1_stop = []       # best 1-stop pit lap per simulation
    optimal_laps_2_stop_pit1 = []  # best 2-stop first pit lap per simulation
    optimal_laps_2_stop_pit2 = []  # best 2-stop second pit lap per simulation
    optimal_compounds_1_stop = []  # best 1-stop compound combo per simulation
    optimal_compounds_2_stop = []  # best 2-stop compound combo per simulation

    # -------------------------------------------------------------------------
    # PARALLEL SIMULATION LOOP
    # Each simulation runs independently in a separate process.
    # pool.imap yields results as they complete, enabling progress tracking.
    # -------------------------------------------------------------------------
    with Pool(processes=12) as pool:
        # Each worker receives (config, unique_seed) to ensure reproducibility
        args = [(config, sim) for sim in range(n_simulations)]
        sim_results = []
        for i, result in enumerate(pool.imap(run_simulation, args)):
            sim_results.append(result)
            # Print progress every 10% of simulations
            if i % (n_simulations // 10) == 0:
                print(f"Simulation {i}/{n_simulations}")
                sys.stdout.flush()

    # -------------------------------------------------------------------------
    # RESULTS PROCESSING
    # Unpack each simulation's results and determine the overall winning strategy
    # -------------------------------------------------------------------------
    for best_time_1, best_time_2, best_1, best_2, lap_times_1_stop, lap_times_2_stop in sim_results:

        # Unpack strategy keys into readable variables
        pit1, c1 = best_1          # e.g. pit1=20, c1=('soft','medium')
        pit2a, pit2b, c2 = best_2  # e.g. pit2a=15, pit2b=38, c2=('soft','medium','hard')

        # Compare best 1-stop vs best 2-stop by total race time
        # min() on tuples compares first element (race time) first
        best = min([(best_time_1, '1-stop', best_1),
                    (best_time_2, '2-stop', best_2)])

        optimal_laps.append(best[1])              # record winning strategy type
        optimal_laps_1_stop.append(pit1)
        optimal_laps_2_stop_pit1.append(pit2a)
        optimal_laps_2_stop_pit2.append(pit2b)
        optimal_compounds_1_stop.append(c1)
        optimal_compounds_2_stop.append(c2)

    # -------------------------------------------------------------------------
    # LAP TRACE: visualize a single race with the most common winning strategy
    # -------------------------------------------------------------------------
    from collections import Counter

    best_1stop_compounds = list(Counter(optimal_compounds_1_stop).most_common(1)[0][0])
    filtered_pit1_1stop = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if list(c) == best_1stop_compounds]
    best_1stop_pit = Counter(filtered_pit1_1stop).most_common(1)[0][0]

    best_2stop_compounds = list(Counter(optimal_compounds_2_stop).most_common(1)[0][0])
    filtered_pit1 = [p1 for p1, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if list(c) == best_2stop_compounds]
    filtered_pit2 = [p2 for p2, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if list(c) == best_2stop_compounds]
    best_2stop_pit1 = Counter(filtered_pit1).most_common(1)[0][0]
    best_2stop_pit2 = Counter(filtered_pit2).most_common(1)[0][0]

    plot_single_race(config, [best_1stop_pit], best_1stop_compounds, filename='lap_trace_1stop.png')
    plot_single_race(config, [best_2stop_pit1, best_2stop_pit2], best_2stop_compounds, filename='lap_trace_2stop.png')

    most_common_strategy = max(set(optimal_laps), key=optimal_laps.count)

    # Print summary statistics
    print(f"Most common optimal pit strategy: {most_common_strategy}")
    print(f"Best 1-stop: pit lap {pit1}, compounds {c1}")
    print(f"Best 2-stop: pit laps {pit2a}/{pit2b}, compounds {c2}")

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------

    # 1-stop compound distribution: which compound combos win most often
    compound_counts_1 = Counter(optimal_compounds_1_stop)
    top_combos = compound_counts_1.most_common(6)
    labels = [f"{c[0][0].upper()}→{c[1][0].upper()}" for c, _ in top_combos]
    values = [v for _, v in top_combos]
    plt.bar(labels, values)
    plt.xlabel("Compound Combination")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Compound Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_medium_v1_compounds_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

    # 2-stop compound distribution
    compound_counts_2 = Counter(optimal_compounds_2_stop)
    top_combos = compound_counts_2.most_common(6)
    labels = [f"{c[0][0].upper()}→{c[1][0].upper()}→{c[2][0].upper()}" for c, _ in top_combos]
    values = [v for _, v in top_combos]
    plt.bar(labels, values)
    plt.xlabel("Compound Combination")
    plt.ylabel("Frequency")
    plt.title(f"Best 2-Stop Compound Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_medium_v1_compounds_2_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

    # 1-stop pit lap distribution: where the optimal pit lap falls across simulations
    plt.hist(optimal_laps_1_stop, bins=range(1, config['total_laps']), edgecolor='black')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps'] / 2), color='red', linestyle='--', label='Deterministic optimum')
    plt.legend()
    plt.savefig('monte_carlo_dist_medium_v1_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

    # 2-stop pit lap distribution: both stops shown together
    plt.hist(optimal_laps_2_stop_pit1, bins=range(1, config['total_laps']),
             edgecolor='black', alpha=0.7, label='First stop')
    plt.hist(optimal_laps_2_stop_pit2, bins=range(1, config['total_laps']),
             edgecolor='black', alpha=0.7, label='Second stop')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps'] / 3), color='red', linestyle='--',
                label='Deterministic optimum, stop 1')
    plt.axvline(x=(2 * config['total_laps'] / 3), color='red', linestyle='--',
                label='Deterministic optimum, stop 2')
    plt.legend()
    plt.savefig('monte_carlo_dist_medium_v1_2_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

    # Overall strategy distribution: how often 1-stop vs 2-stop wins
    counts = Counter(optimal_laps)
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Optimal Strategy")
    plt.ylabel("Frequency")
    plt.title(f"Best Pit Strategy Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_medium_v1_strategy.png', dpi=450, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()