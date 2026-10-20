# F1 Monte Carlo race strategy simulator
#
# Simulates one car's race many times with random safety cars, traffic,
# pit stop times and tire wear, and records which strategy wins each time.
# The result is a spread of good strategies rather than one answer.
#
# Can be run on its own (python monte_carlo.py) or used by streamlit_app.py.

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
from itertools import product

def simulate_race(pit_lap1, compounds, race_events, config, pit_lap2=None, 
                  dnf=False, dnf_lap=None, noise=None, rng=None):
    """
    Run one race lap by lap for a fixed strategy.

    pit_lap1, pit_lap2: laps to pit on (pit_lap2=None means a 1-stop)
    compounds: tire for each stint, e.g. ('soft', 'medium')
    race_events: per-lap event array from generate_race_events
    config: race settings (see main())
    dnf, dnf_lap: if dnf is True the race isn't run and time is inf

    Returns a dict with total_time, lap_times, dnf and dnf_lap.
    """

    if dnf:
        return {'total_time': np.inf, 'lap_times': [], 'dnf': True, 'dnf_lap': dnf_lap}
    
    if noise is None: # lap trace plots dont pass noise in
        rng = rng or np.random.default_rng()
        noise = {'pit': rng.normal(0, 2, config['total_laps']),
                 'deg': rng.normal(0, 0.001, config['total_laps']),
                 'traffic': rng.random(config['total_laps'])}

    total_time = 0
    tire_age = 0       # laps on current tires
    stint_lap = 0      # laps since last stop, used for traffic
    stint_number = 0
    lap_times = []

    pit_laps = [p for p in [pit_lap1, pit_lap2] if p is not None]

    for laps in range(1, config['total_laps'] + 1):

        i = laps - 1

        # drawn every lap but only used if we pit this lap
        pit_loss_randomness = noise['pit'][i]

        tire_degradation_rate_randomness = noise['deg'][i]

        # traffic chance is highest right after a stop and drops off as the field spreads
        traffic_prob = config['base_traffic_prob'] * np.exp(-stint_lap / config['spread_rate'])
        in_traffic = noise['traffic'][i] < traffic_prob

        # number of stops already made gives the current stint
        stint_number = sum(1 for p in pit_laps if laps > p)
        current_compound = compounds[stint_number]
        compound_data = config['compound_data'][current_compound]

        # past max_laps the tire falls off and deg triples
        if tire_age > compound_data['max_laps']:
            deg_this_lap = compound_data['deg_rate'] * 3
        else:
            deg_this_lap = compound_data['deg_rate']

        event = race_events[laps - 1]
        if event == 2:
            current_base = config['sc_base_lap_time']
        elif event == 1:
            current_base = config['vsc_base_lap_time']
        else:
            current_base = config['base_lap_time']

        lap_time = (current_base
                    + compound_data['lap_time_offset']
                    + (deg_this_lap + tire_degradation_rate_randomness) * tire_age)

        # no traffic loss under SC/VSC since everyone is going the same speed
        if in_traffic and event == 0:
            lap_time += config['traffic_time_loss']

        if laps in pit_laps:
            # pitting is cheaper under SC/VSC since the field is slowed down
            if event == 2:
                lap_time += config['sc_pit_loss'] + pit_loss_randomness
            elif event == 1:
                lap_time += config['vsc_pit_loss'] + pit_loss_randomness
            else:
                lap_time += config['pit_loss'] + pit_loss_randomness

            stint_lap = 0
            tire_age = 1
        else:
            tire_age += 1
            stint_lap += 1

        lap_times.append(lap_time)

    total_time = sum(lap_times)
    return {'total_time': total_time, 'lap_times': lap_times, 'dnf': False, 'dnf_lap': None}


def run_simulation(args):
    """
    One Monte Carlo run. Makes a random race (SC/VSC laps, possible DNF),
    then tries strategies against it and keeps the best 1-stop and 2-stop.

    args is (config, seed) so it can be passed straight to a process pool.

    Returns (best_time_1, best_time_2, best_1, best_2,
             lap_times_1_stop, lap_times_2_stop, dnf_count).
    best_1 is (pit_lap, compounds), best_2 is (pit_lap1, pit_lap2, compounds).
    """

    config, seed = args
    rng = np.random.default_rng(seed)
    # same events for every strategy in this run so they're compared fairly
    sc_laps = generate_race_events(config['total_laps'], config['sc_chance'], 
                                   config['vsc_chance'], rng)

    p_dnf = 1 - (1 - config['mechanical_failure_prob']) ** config['total_laps']
    car_dnf = rng.random() < p_dnf
    dnf_count = 1 if car_dnf else 0
    if car_dnf:
        dnf_lap = rng.integers(1, config['total_laps'])
    else:
        dnf_lap = None

    results_1_stop = {}      # (pit_lap, compounds) -> total_time
    results_2_stop = {}      # (pit_lap1, pit_lap2, compounds) -> total_time
    lap_times_1_stop = {}
    lap_times_2_stop = {}

    compound_names = list(config['compound_data'].keys())

    # rules require at least 2 different compounds
    valid_1stop = [(c1, c2) for c1, c2 in product(compound_names, repeat=2) if c1 != c2]
    valid_2stop = [(c1, c2, c3) for c1, c2, c3 in product(compound_names, repeat=3)
                   if len(set([c1, c2, c3])) >= 2]

    noise = {
        'pit': rng.normal(0, 2, config['total_laps']),
        'deg': rng.normal(0, 0.001, config['total_laps']),
        'traffic': rng.random(config['total_laps']),
    }

    if config['search_method'] == 'exhaustive':
        # every pit lap and compound combo, slow but finds the real best
        for pit_lap in range(1, config['total_laps'] - 5):
            for c1stop in valid_1stop:
                result = simulate_race(pit_lap, c1stop, sc_laps, config, None, dnf=car_dnf, dnf_lap=dnf_lap, noise=noise)
                results_1_stop[(pit_lap, c1stop)] = result['total_time']

            for pit_lap2 in range(pit_lap + 1, config['total_laps'] - 2):
                for c2stop in valid_2stop:
                    result = simulate_race(pit_lap, c2stop, sc_laps, config, pit_lap2=pit_lap2, dnf=car_dnf, dnf_lap= dnf_lap, noise=noise)
                    results_2_stop[(pit_lap, pit_lap2, c2stop)] = result['total_time']

    else:
        # random subset of strategies, much faster for the web app
        for pit_lap in range(1, config['total_laps'] - 5):
            for c1stop in valid_1stop:
                result = simulate_race(pit_lap, c1stop, sc_laps, config, None, dnf=car_dnf, dnf_lap=dnf_lap, noise=noise)
                results_1_stop[(pit_lap, c1stop)] = result['total_time']

        for _ in range(config['n_strategies_sampled']):
            pit_lap1 = rng.integers(5, config['total_laps'] - 15)
            pit_lap2 = rng.integers(pit_lap1 + 5, config['total_laps'] - 5)
            compounds = valid_2stop[rng.integers(len(valid_2stop))]
            result = simulate_race(pit_lap1, compounds, sc_laps, config, pit_lap2=pit_lap2, dnf=car_dnf, dnf_lap=dnf_lap, noise=noise)
            results_2_stop[(pit_lap1, pit_lap2, compounds)] = result['total_time']

        # for each compound set, keep moving to the best pit laps within +-3
        # of the current best until nothing nearby is faster
        for comps in valid_2stop:
            cands = [k for k in results_2_stop if k[2] == comps]
            if not cands:
                continue
            p1, p2, _ = min(cands, key=results_2_stop.get)
            while True:
                nearby = []
                for d1 in range(-3, 4):
                    for d2 in range(-3, 4):
                        a, b = p1 + d1, p2 + d2
                        if not (1 <= a < b < config['total_laps']):
                            continue
                        key = (a, b, comps)
                        if key not in results_2_stop:
                            result = simulate_race(a, comps, sc_laps, config, pit_lap2=b, dnf=car_dnf, dnf_lap=dnf_lap, noise=noise)
                            results_2_stop[key] = result['total_time']
                        nearby.append(key)
                best_nearby = min(nearby, key=results_2_stop.get)
                if results_2_stop[best_nearby] >= results_2_stop[(p1, p2, comps)]:
                    break
                p1, p2, _ = best_nearby

    best_1 = min(results_1_stop, key=results_1_stop.get)
    best_2 = min(results_2_stop, key=results_2_stop.get)

    return results_1_stop[best_1], results_2_stop[best_2], best_1, best_2, lap_times_1_stop, lap_times_2_stop, dnf_count


def plot_single_race(config, pit_laps, compounds, filename='lap_trace.png'):
    """Save a lap time plot of one race using the given strategy, with stints shaded by compound."""

    # retry a few times so we don't end up plotting a DNF
    for _ in range(20):
        sc_laps = generate_race_events(config['total_laps'], config['sc_chance'], config['vsc_chance'])
        result = simulate_race(pit_laps[0], compounds, sc_laps, config,
                            pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)
        if not result['dnf']:
            break
    lap_times = result['lap_times']

    compound_colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'white'}

    stint_boundaries = [0] + pit_laps + [config['total_laps']]

    plt.plot(range(1, config['total_laps'] + 1), lap_times)

    for pit in pit_laps:
        plt.axvline(x=pit, color='red', linestyle='--', label=f'Pit lap {pit}')

    for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
        plt.axvspan(start, end, alpha=0.2, color=compound_colors[compounds[i]],
                    label=f'Stint {i + 1}: {compounds[i]}')

    plt.xlabel('Lap')
    plt.ylabel('Lap time (s)')
    plt.title('Single race lap time trace')
    plt.legend()
    plt.ylim(config['base_lap_time'] - 2, config['base_lap_time'] + config['pit_loss'] + 10)
    plt.savefig(filename, dpi=450, bbox_inches='tight')
    plt.close()


def generate_race_events(total_laps, sc_chance, vsc_chance, rng=None):
    """
    Random event for each lap: 0 = green, 1 = VSC, 2 = SC.
    Only one event per lap.
    """
    rng = rng or np.random.default_rng()
    events = np.zeros(total_laps, dtype=int)

    for lap in range(total_laps):
        if rng.random() < sc_chance:
            events[lap] = 2 #SC
        elif rng.random() < vsc_chance:
            events[lap] = 1 #VSC

    return events


def main():

    config = {
        'base_lap_time': 81,          # medium tire, no deg (s)
        'pit_loss': 25,               # green flag pit loss (s)
        'sc_pit_loss': 12,
        'sc_base_lap_time': 90,
        'vsc_pit_loss': 18,
        'vsc_base_lap_time': 87,
        'mechanical_failure_prob': 1/200,   # per lap
        'total_laps': 60,

        # per-race % spread over the laps, so 50 here is ~50% chance of an SC in the race
        'sc_chance': 50 / 60 / 100,
        'vsc_chance': 30 / 60 / 100,

        'base_traffic_prob': 0.4,    # traffic chance on the first lap of a stint
        'spread_rate': 15,           # higher = traffic clears slower
        'traffic_time_loss': 0.5,    # s per lap stuck in traffic

        # offset is vs medium (negative = faster), deg is s per lap of tire age,
        # max_laps is where the cliff starts
        'compound_data': {
            'soft':   {'lap_time_offset': -0.6, 'deg_rate': 0.08, 'max_laps': 15},
            'medium': {'lap_time_offset':  0.0, 'deg_rate': 0.05, 'max_laps': 35},
            'hard':   {'lap_time_offset':  0.6, 'deg_rate': 0.02, 'max_laps': 50},
        },

        'search_method': 'sampled',   # or 'exhaustive'
        'n_strategies_sampled': 500,
    }

    n_simulations = 1000

    optimal_laps = []              # '1-stop' or '2-stop', whichever won
    optimal_laps_1_stop = []
    optimal_laps_2_stop_pit1 = []
    optimal_laps_2_stop_pit2 = []
    optimal_compounds_1_stop = []
    optimal_compounds_2_stop = []
    winning_strategies_1_stop = []
    winning_strategies_2_stop = []
    best_times_1_stop = []
    best_times_2_stop = []

    with Pool(processes=12) as pool:
        args = [(config, sim) for sim in range(n_simulations)]
        sim_results = []
        for i, result in enumerate(pool.imap(run_simulation, args)):
            sim_results.append(result)
            if i % (n_simulations // 10) == 0:
                print(f"Simulation {i}/{n_simulations}")
                sys.stdout.flush()

    total_dnfs = 0
    for best_time_1, best_time_2, best_1, best_2, lap_times_1_stop, lap_times_2_stop, dnf_count in sim_results:
        total_dnfs += dnf_count

        if dnf_count:
            continue

        pit1, c1 = best_1
        pit2a, pit2b, c2 = best_2

        # tuples compare on time first
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
            winning_strategies_1_stop.append((pit1, c1))
        else:
            winning_strategies_2_stop.append((pit2a, pit2b, c2))

    # lap traces for the most common compounds and pit laps
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

    avg_time_1stop = np.mean(best_times_1_stop)
    avg_time_2stop = np.mean(best_times_2_stop)
    most_common_strategy = max(set(optimal_laps), key=optimal_laps.count)

    n_finished = n_simulations - total_dnfs

    print(f"Most common optimal pit strategy: {most_common_strategy}")
    print(f"1-stop wins: {optimal_laps.count('1-stop')}/{n_finished} ({optimal_laps.count('1-stop')/n_finished*100:.1f}%)")
    print(f"2-stop wins: {optimal_laps.count('2-stop')}/{n_finished} ({optimal_laps.count('2-stop')/n_finished*100:.1f}%)")
    print(f"Avg best 1-stop time: {avg_time_1stop:.1f}s ({avg_time_1stop//60:.0f}m {avg_time_1stop%60:.1f}s)")
    print(f"Avg best 2-stop time: {avg_time_2stop:.1f}s ({avg_time_2stop//60:.0f}m {avg_time_2stop%60:.1f}s)")
    print(f"Best 1-stop: pit lap {best_1stop_pit}, compounds {' → '.join([c[0].upper() for c in best_1stop_compounds])}")
    print(f"Best 2-stop: pit laps {best_2stop_pit1}/{best_2stop_pit2}, compounds {' → '.join([c[0].upper() for c in best_2stop_compounds])}")
    print(f"Total DNFs: {total_dnfs} ({total_dnfs/n_simulations*100:.1f}% of simulations)")

    # plots

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

    plt.hist(optimal_laps_1_stop, bins=range(1, config['total_laps']), edgecolor='black')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps'] / 2), color='red', linestyle='--', label='Deterministic optimum')
    plt.legend()
    plt.savefig('monte_carlo_dist_medium_v1_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

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

    # pit laps split by compound, explains the double peaks in the histograms
    bins = range(1, config['total_laps'])

    soft_1stop  = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'soft']
    med_1stop   = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'medium']
    hard_1stop  = [p for p, c in zip(optimal_laps_1_stop, optimal_compounds_1_stop) if c[0] == 'hard']
    soft_pit1   = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'soft']
    med_pit1    = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'medium']
    hard_pit1   = [p for p, c in zip(optimal_laps_2_stop_pit1, optimal_compounds_2_stop) if c[0] == 'hard']
    soft_pit2   = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'soft']
    med_pit2    = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'medium']
    hard_pit2   = [p for p, c in zip(optimal_laps_2_stop_pit2, optimal_compounds_2_stop) if c[1] == 'hard']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.hist([soft_1stop, med_1stop, hard_1stop], bins=bins, stacked=True,
            color=['red', 'gold', 'gray'], label=['S', 'M', 'H'], edgecolor='black', linewidth=0.3)
    ax1.set_xlabel("Optimal Pit Lap")
    ax1.set_ylabel("Frequency")
    ax1.set_title("1-Stop: opening compound")
    ax1.legend()

    ax2.hist([soft_pit1, med_pit1, hard_pit1], bins=bins, stacked=True,
            color=['red', 'gold', 'gray'], label=['S', 'M', 'H'], edgecolor='black', linewidth=0.3)
    ax2.set_xlabel("Optimal Pit Lap")
    ax2.set_ylabel("Frequency")
    ax2.set_title("2-Stop First Stop: opening compound")
    ax2.legend()

    ax3.hist([soft_pit2, med_pit2, hard_pit2], bins=bins, stacked=True,
            color=['red', 'gold', 'gray'], label=['S', 'M', 'H'], edgecolor='black', linewidth=0.3)
    ax3.set_xlabel("Optimal Pit Lap")
    ax3.set_ylabel("Frequency")
    ax3.set_title("2-Stop Second Stop: second stint compound")
    ax3.legend()

    plt.tight_layout()
    plt.savefig('pit_lap_by_compound.png', dpi=450, bbox_inches='tight')
    plt.close()

    counts = Counter(optimal_laps)
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Optimal Strategy")
    plt.ylabel("Frequency")
    plt.title(f"Best Pit Strategy Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_medium_v1_strategy.png', dpi=450, bbox_inches='tight')
    plt.close()

    print("\n--- Top 10 1-Stop Strategies ---")
    pit_lap_counts_1 = Counter(winning_strategies_1_stop)
    total_1stop_wins = len(winning_strategies_1_stop)
    for i, ((pit, comp), freq) in enumerate(pit_lap_counts_1.most_common(10), 1):
        print(f"{i:2}. Start: {comp[0][0].upper()} | Stop 1: Lap {pit} → {comp[1][0].upper()} | "
            f"Freq: {freq} | Win Rate: {freq/total_1stop_wins*100:.1f}%")

    print("\n--- Top 10 2-Stop Strategies ---")
    pit_lap_counts_2 = Counter(winning_strategies_2_stop)
    total_2stop_wins = len(winning_strategies_2_stop)
    for i, ((pit1, pit2, comp), freq) in enumerate(pit_lap_counts_2.most_common(10), 1):
        print(f"{i:2}. Start: {comp[0][0].upper()} | Stop 1: Lap {pit1} → {comp[1][0].upper()} | "
            f"Stop 2: Lap {pit2} → {comp[2][0].upper()} | Freq: {freq} | Win Rate: {freq/total_2stop_wins*100:.1f}%")


if __name__ == "__main__":
    main()
