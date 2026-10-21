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
from collections import Counter, defaultdict

def make_noise(total_laps, rng):
    """Per-lap random draws for pit stop time, deg and traffic."""
    return {'pit': rng.normal(0, 2, total_laps),
            'deg': rng.normal(0, 0.001, total_laps),
            'traffic': rng.random(total_laps)}


def lap_times_batch(strategies, race_events, config, noise):
    """
    Lap times for many strategies on the same race at once.

    strategies: list of (pit_lap1, pit_lap2, compounds), pit_lap2=None for a 1-stop
    Returns an array of shape (len(strategies), total_laps).

    Each row is worked out the same way as running the race lap by lap:
    tire wear resets to 1 lap's worth the lap after a stop, and the pit lap
    itself still counts as part of the old stint.

    Tires wear by 1 lap's worth per lap, plus extra on a heavy fuel load
    (fuel_wear) and when stuck in traffic (dirty_air_wear). Deg and the
    cliff work off that wear rather than plain laps.
    """
    total_laps = config['total_laps']
    names = list(config['compound_data'])
    data = config['compound_data']
    offsets = np.array([data[c]['lap_time_offset'] for c in names])
    degs = np.array([data[c]['deg_rate'] for c in names])
    max_laps = np.array([data[c]['max_laps'] for c in names])

    no_stop = total_laps + 1   # stands in for a missing second stop
    pit1 = np.array([s[0] for s in strategies])[:, None]
    pit2 = np.array([no_stop if s[1] is None else s[1] for s in strategies])[:, None]
    # pad 1-stops to 3 stints, the third is never reached
    comps = np.array([[names.index(c) for c in (s[2] if len(s[2]) == 3 else s[2] + s[2][-1:])]
                      for s in strategies])

    lap = np.arange(1, total_laps + 1)
    stint = (lap > pit1).astype(int) + (lap > pit2)
    last_pit = np.where(lap > pit2, pit2, np.where(lap > pit1, pit1, 0))
    stint_lap = np.where(last_pit > 0, lap - last_pit - 1, lap - 1)
    events = np.asarray(race_events)

    # traffic chance is highest right after a stop and drops off as the field spreads.
    # no traffic under SC/VSC since everyone is going the same speed
    traffic_prob = config['base_traffic_prob'] * np.exp(-stint_lap / config['spread_rate'])
    in_traffic = (noise['traffic'] < traffic_prob) & (events == 0)

    # wear added each lap: 1 lap's worth, more on a full tank and in dirty air
    fuel_left = (total_laps - lap) / (total_laps - 1)   # 1 on lap 1, 0 on the last lap
    wear_step = (1 + config.get('fuel_wear', 0) * fuel_left
                 + config.get('dirty_air_wear', 0) * in_traffic)
    wear_after = np.cumsum(wear_step, axis=1)   # since the start of the race
    wear_before = wear_after - wear_step
    wear_at_stop = np.take_along_axis(wear_after, np.maximum(last_pit - 1, 0), axis=1)
    wear = np.where(last_pit > 0, 1 + wear_before - wear_at_stop, wear_before)

    c = np.take_along_axis(comps, stint, axis=1)
    # past max_laps of wear the tire falls off and deg triples
    deg = np.where(wear > max_laps[c], degs[c] * 3, degs[c])

    base = np.where(events == 2, config['sc_base_lap_time'],
                    np.where(events == 1, config['vsc_base_lap_time'], config['base_lap_time']))
    times = base + offsets[c] + (deg + noise['deg']) * wear

    # heavier car is slower, only on green laps since SC/VSC set the pace
    fuel_pace = config.get('fuel_effect', 0) * (total_laps - lap)
    times = np.where(events == 0, times + fuel_pace, times)
    times = np.where(in_traffic, times + config['traffic_time_loss'], times)

    # pitting is cheaper under SC/VSC since the field is slowed down
    pit_cost = np.where(events == 2, config['sc_pit_loss'],
                        np.where(events == 1, config['vsc_pit_loss'], config['pit_loss'])) + noise['pit']
    pitting = (lap == pit1) | (lap == pit2)
    return np.where(pitting, times + pit_cost, times)


def race_totals(strategies, race_events, config, noise, dnf=False):
    """Total race time for each strategy, inf for all of them if the car DNFs."""
    if dnf:
        return np.full(len(strategies), np.inf)
    # cumsum adds left to right like a plain loop would
    return np.cumsum(lap_times_batch(strategies, race_events, config, noise), axis=1)[:, -1]


def simulate_race(pit_lap1, compounds, race_events, config, pit_lap2=None,
                  dnf=False, dnf_lap=None, noise=None, rng=None):
    """
    Run one race for a fixed strategy.

    pit_lap1, pit_lap2: laps to pit on (pit_lap2=None means a 1-stop)
    compounds: tire for each stint, e.g. ('soft', 'medium')
    race_events: per-lap event array from generate_race_events
    config: race settings (see main())
    dnf, dnf_lap: if dnf is True the race isn't run and time is inf
    noise: per-lap draws from make_noise, made fresh if not given

    Returns a dict with total_time, lap_times, dnf and dnf_lap.
    """

    if dnf:
        return {'total_time': np.inf, 'lap_times': [], 'dnf': True, 'dnf_lap': dnf_lap}

    if noise is None:  # lap trace plots don't pass noise in
        noise = make_noise(config['total_laps'], rng or np.random.default_rng())

    lap_times = lap_times_batch([(pit_lap1, pit_lap2, tuple(compounds))], race_events, config, noise)[0]
    return {'total_time': sum(lap_times.tolist()), 'lap_times': lap_times.tolist(), 'dnf': False, 'dnf_lap': None}


def valid_compounds(config):
    """1-stop and 2-stop compound combos, the rules require at least 2 different compounds."""
    names = list(config['compound_data'].keys())
    valid_1stop = [(c1, c2) for c1, c2 in product(names, repeat=2) if c1 != c2]
    valid_2stop = [(c1, c2, c3) for c1, c2, c3 in product(names, repeat=3)
                   if len(set([c1, c2, c3])) >= 2]
    return valid_1stop, valid_2stop


def one_stop_keys(total_laps, valid_1stop):
    """Every 1-stop searched: (pit_lap, compounds), stopping no later than 6 laps from the end."""
    return [(p, c) for p in range(1, total_laps - 5) for c in valid_1stop]


def two_stop_keys(total_laps, valid_2stop):
    """Every 2-stop searched exhaustively: (pit_lap1, pit_lap2, compounds)."""
    return [(p1, p2, c) for p1 in range(1, total_laps - 5)
            for p2 in range(p1 + 1, total_laps - 2) for c in valid_2stop]


def calm_optimum(config):
    """
    Best 1-stop and 2-stop on a race with no SC/VSC, traffic or noise,
    searching every strategy. A reference point for the histograms.

    Returns (best_1, best_2) in the same form as run_simulation.
    """
    laps = config['total_laps']
    green = np.zeros(laps, dtype=int)
    calm = {'pit': np.zeros(laps), 'deg': np.zeros(laps), 'traffic': np.ones(laps)}
    valid_1stop, valid_2stop = valid_compounds(config)

    ones = one_stop_keys(laps, valid_1stop)
    twos = two_stop_keys(laps, valid_2stop)
    t1 = race_totals([(p, None, c) for p, c in ones], green, config, calm)
    t2 = race_totals(twos, green, config, calm)
    return ones[int(np.argmin(t1))], twos[int(np.argmin(t2))]


def run_simulation(args):
    """
    One Monte Carlo run. Makes a random race (SC/VSC laps, possible DNF),
    then tries strategies against it and keeps the best 1-stop and 2-stop.

    args is (config, seed) so it can be passed straight to a process pool.

    Returns (best_time_1, best_time_2, best_1, best_2, dnf_count).
    best_1 is (pit_lap, compounds), best_2 is (pit_lap1, pit_lap2, compounds).
    """

    config, seed = args
    if config['total_laps'] < 10:
        raise ValueError("races need at least 10 laps")
    rng = np.random.default_rng(seed)
    # same events for every strategy in this run so they're compared fairly
    sc_laps = generate_race_events(config['total_laps'], config['sc_chance'], 
                                   config['vsc_chance'], rng)

    car_dnf = rng.random() < dnf_rate(config['mechanical_failure_prob'], config['total_laps'])
    dnf_count = 1 if car_dnf else 0
    if car_dnf:
        dnf_lap = rng.integers(1, config['total_laps'])
    else:
        dnf_lap = None

    results_1_stop = {}      # (pit_lap, compounds) -> total_time
    results_2_stop = {}      # (pit_lap1, pit_lap2, compounds) -> total_time

    def run_2stops(keys):
        # times any 2-stops we haven't tried yet, all in one batch
        new = [k for k in dict.fromkeys(keys) if k not in results_2_stop]
        if new:
            totals = race_totals([(p1, p2, comps) for p1, p2, comps in new], sc_laps, config, noise, car_dnf)
            results_2_stop.update(zip(new, totals.tolist()))

    valid_1stop, valid_2stop = valid_compounds(config)

    noise = make_noise(config['total_laps'], rng)

    # every 1-stop in both modes, it's only a few hundred races
    keys_1 = one_stop_keys(config['total_laps'], valid_1stop)
    totals = race_totals([(p, None, c) for p, c in keys_1], sc_laps, config, noise, car_dnf)
    results_1_stop.update(zip(keys_1, totals.tolist()))

    if config['search_method'] == 'exhaustive':
        # every 2-stop too, slow but finds the real best
        run_2stops(two_stop_keys(config['total_laps'], valid_2stop))

    else:
        # random 2-stops, then refine
        sampled = []
        laps = config['total_laps']
        for _ in range(config['n_strategies_sampled']):
            # keep stops away from the start and end, unless the race is too
            # short for that, then anywhere with the second stop after the first
            low, high = (5, laps - 15) if laps - 15 > 5 else (1, laps - 2)
            pit_lap1 = int(rng.integers(low, high))
            low, high = (pit_lap1 + 5, laps - 5) if laps - 5 > pit_lap1 + 5 else (pit_lap1 + 1, laps)
            pit_lap2 = int(rng.integers(low, high))
            sampled.append((pit_lap1, pit_lap2, valid_2stop[rng.integers(len(valid_2stop))]))
        run_2stops(sampled)

        # for each compound set, keep moving to the best pit laps within +-3
        # of the current best until nothing nearby is faster
        for comps in valid_2stop:
            cands = [k for k in results_2_stop if k[2] == comps]
            if not cands:
                continue
            p1, p2, _ = min(cands, key=results_2_stop.get)
            while True:
                nearby = [(p1 + d1, p2 + d2, comps) for d1 in range(-3, 4) for d2 in range(-3, 4)
                          if 1 <= p1 + d1 < p2 + d2 < config['total_laps']]
                run_2stops(nearby)
                best_nearby = min(nearby, key=results_2_stop.get)
                if results_2_stop[best_nearby] >= results_2_stop[(p1, p2, comps)]:
                    break
                p1, p2, _ = best_nearby

    best_1 = min(results_1_stop, key=results_1_stop.get)
    best_2 = min(results_2_stop, key=results_2_stop.get)

    return results_1_stop[best_1], results_2_stop[best_2], best_1, best_2, dnf_count


def summarize(sim_results, total_laps):
    """
    Collect the results of many run_simulation calls into the numbers
    the app and main() show. DNF runs are counted then left out.
    """
    s = {'n_sims': len(sim_results), 'total_dnfs': 0, 'winner': [],
         'pit_1': [], 'comps_1': [], 'pit_2a': [], 'pit_2b': [], 'comps_2': [],
         'times_1': [], 'times_2': []}
    wins_1 = defaultdict(list)   # winning strategy -> race times, only runs where it beat the other type
    wins_2 = defaultdict(list)

    for best_time_1, best_time_2, best_1, best_2, dnf_count in sim_results:
        s['total_dnfs'] += dnf_count
        if dnf_count:
            continue
        pit1, c1 = best_1
        pit2a, pit2b, c2 = best_2
        s['pit_1'].append(pit1)
        s['comps_1'].append(c1)
        s['pit_2a'].append(pit2a)
        s['pit_2b'].append(pit2b)
        s['comps_2'].append(c2)
        s['times_1'].append(best_time_1)
        s['times_2'].append(best_time_2)
        if best_time_1 <= best_time_2:
            s['winner'].append('1-stop')
            wins_1[best_1].append(best_time_1)
        else:
            s['winner'].append('2-stop')
            wins_2[best_2].append(best_time_2)

    s['n_finished'] = s['n_sims'] - s['total_dnfs']
    if s['n_finished'] == 0:
        raise ValueError("every simulation ended in a DNF, lower the failure probability")

    s['wins_1'] = s['winner'].count('1-stop')
    s['wins_2'] = s['winner'].count('2-stop')
    s['most_common'] = '1-stop' if s['wins_1'] >= s['wins_2'] else '2-stop'
    s['avg_time_1'] = float(np.mean(s['times_1']))
    s['avg_time_2'] = float(np.mean(s['times_2']))

    # top 10 winning strategies of each type, (strategy, times it won)
    s['top_1'] = Counter({k: len(v) for k, v in wins_1.items()}).most_common(10)
    s['top_2'] = Counter({k: len(v) for k, v in wins_2.items()}).most_common(10)

    # pit laps split by starting compound (second stint for the 2nd stop),
    # explains the double peaks in the pit lap histograms
    s['by_compound'] = {}
    for name, pits, comps, stint in [('1-stop', s['pit_1'], s['comps_1'], 0),
                                     ('2-stop first', s['pit_2a'], s['comps_2'], 0),
                                     ('2-stop second', s['pit_2b'], s['comps_2'], 1)]:
        s['by_compound'][name] = {tire: [p for p, c in zip(pits, comps) if c[stint] == tire]
                                  for tire in ('soft', 'medium', 'hard')}

    # strategy to show in the lap traces: #1 in the table, ties go to the faster
    # average time. falls back to an even split if that type never won
    def representative(wins, all_comps, fallback_pits):
        if not wins:
            return fallback_pits, list(Counter(all_comps).most_common(1)[0][0])
        top_freq = max(len(v) for v in wins.values())
        top = min((k for k, v in wins.items() if len(v) == top_freq), key=lambda k: np.mean(wins[k]))
        return list(top[:-1]), list(top[-1])

    s['rep_1'] = representative(wins_1, s['comps_1'], [total_laps // 2])
    s['rep_2'] = representative(wins_2, s['comps_2'], [total_laps // 3, 2 * total_laps // 3])
    return s


def dnf_rate(failure_prob, total_laps):
    """Chance a car fails at some point in the race, from the per-lap chance."""
    return 1 - (1 - failure_prob) ** total_laps


def sims_needed(margin=0.03, dnf=0.0, z=1.96):
    """
    Simulations needed for the 1-stop vs 2-stop win share to be within
    +-margin (95% confidence for z=1.96). Uses the worst case, an even split,
    and adds enough extra runs to cover the ones lost to DNFs. The search
    method doesn't change this, only how long each run takes.
    """
    finished = (z / margin) ** 2 * 0.25
    return int(np.ceil(finished / (1 - dnf)))


def win_share_margin(wins, finished, z=1.96):
    """
    +- on a win share of wins / finished at 95% confidence. Wilson interval,
    so it stays sensible when one strategy wins almost every race.
    """
    if finished == 0:
        return 1.0
    p = wins / finished
    return float(z / (1 + z ** 2 / finished)
                 * np.sqrt(p * (1 - p) / finished + z ** 2 / (4 * finished ** 2)))


def compound_label(comps):
    """('soft', 'medium') -> 'S → M'"""
    return ' → '.join(c[0].upper() for c in comps)


def trace_race(config, pit_laps, compounds):
    """
    Lap times for one fresh random race using the given strategy, for the lap
    trace plots. No DNF is drawn, so the trace is always a full race.
    """
    events = generate_race_events(config['total_laps'], config['sc_chance'], config['vsc_chance'])
    return simulate_race(pit_laps[0], compounds, events, config,
                         pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)['lap_times']


def plot_single_race(config, pit_laps, compounds, filename='lap_trace.png'):
    """Save a lap time plot of one race using the given strategy, with stints shaded by compound."""

    lap_times = trace_race(config, pit_laps, compounds)

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
        'mechanical_failure_prob': 0.002,   # per lap, ~11% of races over 60 laps
        'total_laps': 60,

        # per-race % spread over the laps, so 50 here is ~50% chance of an SC in the race
        'sc_chance': 50 / 60 / 100,
        'vsc_chance': 30 / 60 / 100,

        'base_traffic_prob': 0.4,    # traffic chance on the first lap of a stint
        'spread_rate': 15,           # higher = traffic clears slower
        'traffic_time_loss': 0.5,    # s per lap stuck in traffic
        'dirty_air_wear': 0.5,       # extra laps of tire wear per lap in traffic

        'fuel_effect': 0.035,        # s per lap slower for each lap of fuel on board, fitted from 2026 races
        'fuel_wear': 0.3,            # extra tire wear on a full tank, fades to 0 by the end

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

    with Pool() as pool:   # one worker per CPU core
        args = [(config, sim) for sim in range(n_simulations)]
        sim_results = []
        for i, result in enumerate(pool.imap(run_simulation, args)):
            sim_results.append(result)
            if i % max(1, n_simulations // 10) == 0:
                print(f"Simulation {i}/{n_simulations}")
                sys.stdout.flush()

    s = summarize(sim_results, config['total_laps'])
    calm_1, calm_2 = calm_optimum(config)
    n_finished = s['n_finished']

    (rep_1_pits, rep_1_comps), (rep_2_pits, rep_2_comps) = s['rep_1'], s['rep_2']
    plot_single_race(config, rep_1_pits, rep_1_comps, filename='lap_trace_1stop.png')
    plot_single_race(config, rep_2_pits, rep_2_comps, filename='lap_trace_2stop.png')

    print(f"Most common optimal pit strategy: {s['most_common']}")
    print(f"1-stop wins: {s['wins_1']}/{n_finished} ({s['wins_1']/n_finished*100:.1f}%)")
    print(f"2-stop wins: {s['wins_2']}/{n_finished} ({s['wins_2']/n_finished*100:.1f}%)")
    for label, t in [('1-stop', s['avg_time_1']), ('2-stop', s['avg_time_2'])]:
        print(f"Avg best {label} time: {t:.1f}s ({t//60:.0f}m {t%60:.1f}s)")
    print(f"Best 1-stop: pit lap {rep_1_pits[0]}, compounds {compound_label(rep_1_comps)}")
    print(f"Best 2-stop: pit laps {rep_2_pits[0]}/{rep_2_pits[1]}, compounds {compound_label(rep_2_comps)}")
    print(f"Total DNFs: {s['total_dnfs']} ({s['total_dnfs']/n_simulations*100:.1f}% of simulations)")

    # plots
    title_end = f"({n_finished:,} finished races)"

    for comps, name in [(s['comps_1'], '1'), (s['comps_2'], '2')]:
        top_combos = Counter(comps).most_common(6)
        plt.bar([compound_label(c) for c, _ in top_combos], [v for _, v in top_combos])
        plt.xlabel("Compound Combination")
        plt.ylabel("Frequency")
        plt.title(f"Best {name}-Stop Compound Distribution {title_end}")
        plt.savefig(f'monte_carlo_dist_medium_v1_compounds_{name}_stop.png', dpi=450, bbox_inches='tight')
        plt.close()

    bins = range(1, config['total_laps'])

    plt.hist(s['pit_1'], bins=bins, edgecolor='black')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Pit Lap Distribution {title_end}")
    plt.axvline(x=calm_1[0], color='red', linestyle='--', label='Best with no SC, traffic or noise')
    plt.legend()
    plt.savefig('monte_carlo_dist_medium_v1_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

    plt.hist(s['pit_2a'], bins=bins, edgecolor='black', alpha=0.7, label='First stop')
    plt.hist(s['pit_2b'], bins=bins, edgecolor='black', alpha=0.7, label='Second stop')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 2-Stop Pit Lap Distribution {title_end}")
    plt.axvline(x=calm_2[0], color='red', linestyle='--', label='Best with no SC, traffic or noise, stop 1')
    plt.axvline(x=calm_2[1], color='orange', linestyle='--', label='Best with no SC, traffic or noise, stop 2')
    plt.legend()
    plt.savefig('monte_carlo_dist_medium_v1_2_stop.png', dpi=450, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = {'1-stop': "1-Stop: opening compound",
              '2-stop first': "2-Stop First Stop: opening compound",
              '2-stop second': "2-Stop Second Stop: second stint compound"}
    for ax, (name, split) in zip(axes, s['by_compound'].items()):
        ax.hist([split['soft'], split['medium'], split['hard']], bins=bins, stacked=True,
                color=['red', 'gold', 'gray'], label=['S', 'M', 'H'], edgecolor='black', linewidth=0.3)
        ax.set_xlabel("Optimal Pit Lap")
        ax.set_ylabel("Frequency")
        ax.set_title(titles[name])
        ax.legend()
    plt.tight_layout()
    plt.savefig('pit_lap_by_compound.png', dpi=450, bbox_inches='tight')
    plt.close()

    counts = Counter(s['winner'])
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Optimal Strategy")
    plt.ylabel("Frequency")
    plt.title(f"Best Pit Strategy Distribution {title_end}")
    plt.savefig('monte_carlo_dist_medium_v1_strategy.png', dpi=450, bbox_inches='tight')
    plt.close()

    print("\n--- Top 10 1-Stop Strategies ---")
    for i, ((pit, comp), freq) in enumerate(s['top_1'], 1):
        print(f"{i:2}. Start: {comp[0][0].upper()} | Stop 1: Lap {pit} → {comp[1][0].upper()} | "
            f"Freq: {freq} | Win Rate: {freq/s['wins_1']*100:.1f}%")

    print("\n--- Top 10 2-Stop Strategies ---")
    for i, ((pit1, pit2, comp), freq) in enumerate(s['top_2'], 1):
        print(f"{i:2}. Start: {comp[0][0].upper()} | Stop 1: Lap {pit1} → {comp[1][0].upper()} | "
            f"Stop 2: Lap {pit2} → {comp[2][0].upper()} | Freq: {freq} | Win Rate: {freq/s['wins_2']*100:.1f}%")


if __name__ == "__main__":
    main()
