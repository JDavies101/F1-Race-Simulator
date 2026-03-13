# introduce multiple tire compounds
# introduce sampled or exhaustive mode

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool
import sys
from itertools import product
import random

def simulate_race(pit_lap1, compounds, sc_laps, config, pit_lap2=None):
    
    total_time = 0
    tire_age = 0
    stint_lap = 0
    stint_number = 0
    lap_times = []
    
    pit_laps = [p for p in [pit_lap1, pit_lap2] if p is not None]
    
    for laps in range(1, config['total_laps']+1):
        
        pit_loss_randomness = np.random.normal(0, 2)
        tire_degradation_rate_randomness = np.random.normal(0, 0.001)    
           
        traffic_prob = config['base_traffic_prob'] * np.exp(-stint_lap / config['spread_rate'])
        in_traffic = np.random.random() < traffic_prob
        
        stint_number = sum(1 for p in pit_laps if laps > p)
        current_compound = compounds[stint_number]  
        compound_data = config['compound_data'][current_compound]
        
        if tire_age > compound_data['max_laps']:
            deg_this_lap = compound_data['deg_rate'] * 3  # cliff multiplier
        else:
            deg_this_lap = compound_data['deg_rate']
        
        if sc_laps[laps - 1]:
            current_base = config['sc_base_lap_time']
        else:
            current_base = config['base_lap_time']

        lap_time = (current_base 
                    + compound_data['lap_time_offset']
                    + (deg_this_lap + tire_degradation_rate_randomness) * tire_age)
        
        if in_traffic and not sc_laps[laps - 1]:
            lap_time += config['traffic_time_loss']
            
        if laps in pit_laps:
            if sc_laps[laps - 1]:
                lap_time += config['sc_pit_loss'] + pit_loss_randomness
            else:
                lap_time += config['pit_loss'] + pit_loss_randomness
    
            stint_lap = 0
            tire_age = 1
        else:
            tire_age += 1
            stint_lap += 1
        
        lap_times.append(lap_time)
        
    total_time = sum(lap_times)
    
    return total_time, lap_times

def run_simulation(args):
    config, seed = args
    np.random.seed(seed)
    
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
    results_1_stop = {}
    results_2_stop = {}
    lap_times_1_stop = {}
    lap_times_2_stop = {}
    
    compound_names = list(config['compound_data'].keys())
    valid_1stop = [(c1, c2) for c1, c2 in product(compound_names, repeat=2) if c1 != c2]
    valid_2stop = [(c1, c2, c3) for c1, c2, c3 in product(compound_names, repeat=3) 
                if len(set([c1, c2, c3])) >= 2]    
    
    if config['search_method'] == 'exhaustive':
        for pit_lap in range(1, config['total_laps']-5):
            for c1stop in valid_1stop:
                results_1_stop[(pit_lap, c1stop)], lap_times_1_stop[(pit_lap, c1stop)] = simulate_race(
                    pit_lap, c1stop, sc_laps, config)
                
            for pit_lap2 in range(pit_lap + 1, config['total_laps']-2):
                for c2stop in valid_2stop:
                    results_2_stop[(pit_lap, pit_lap2, c2stop)], _ = simulate_race(
                        pit_lap, c2stop, sc_laps, config, pit_lap2=pit_lap2)
    
    else:
    # random sampling loop
        for _ in range(config['n_strategies_sampled']):
            pit_lap1 = np.random.randint(5, config['total_laps']-15)
            pit_lap2 = np.random.randint(pit_lap1+5, config['total_laps']-5)
            n_stops = np.random.choice([1, 2])
            if n_stops == 1:
                compounds = random.choice(valid_1stop)
                results_1_stop[(pit_lap1, compounds)], _ = simulate_race(
                    pit_lap1, compounds, sc_laps, config)
            else:
                compounds = random.choice(valid_2stop)
                results_2_stop[(pit_lap1, pit_lap2, compounds)], _ = simulate_race(
                        pit_lap1, compounds, sc_laps, config, pit_lap2=pit_lap2)
    
    best_1 = min(results_1_stop, key=results_1_stop.get) if results_1_stop else None
    best_2 = min(results_2_stop, key=results_2_stop.get) if results_2_stop else None
    
    return results_1_stop[best_1], results_2_stop[best_2], best_1, best_2, lap_times_1_stop, lap_times_2_stop

def plot_single_race(config, pit_laps, compounds):
        sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
        
        total_time, lap_times = simulate_race(
                    pit_laps[0], compounds, sc_laps, config,
                    pit_lap2=pit_laps[1] if len(pit_laps) > 1 else None)
        
        compound_colors = {'soft': 'red', 'medium': 'yellow', 'hard': 'white'}
        stint_boundaries = [0] + pit_laps + [config['total_laps']]
        
        plt.plot(range(1, config['total_laps']+1), lap_times)
        # mark pit stops
        for pit in pit_laps:
            plt.axvline(x=pit, color='red', linestyle='--', label=f'Pit lap {pit}')
        plt.xlabel('Lap')
        plt.ylabel('Lap time (s)')
        plt.title('Single race lap time trace')
        for i, (start, end) in enumerate(zip(stint_boundaries, stint_boundaries[1:])):
            plt.axvspan(start, end, alpha=0.2, color=compound_colors[compounds[i]], 
                        label=f'Stint {i+1}: {compounds[i]}')
        plt.legend()
        plt.ylim(config['base_lap_time']-2, config['base_lap_time']+config['pit_loss']+10)
        plt.savefig('lap_trace_v7.png', dpi=450, bbox_inches='tight')
        plt.close()

def generate_safety_cars(total_laps, sc_chance):
    
    return np.random.random(total_laps) < sc_chance

def main():
    
    config = {
        'base_lap_time': 81,
        'pit_loss': 25,
        'sc_pit_loss': 12,
        'sc_base_lap_time': 90,
        'total_laps': 60,
        'sc_chance': 50/60/100, # 50% chance per race
        'base_traffic_prob': 0.4,
        'spread_rate': 15, 
        'traffic_time_loss': 0.5,
        'compound_data': {
            'soft':   {'lap_time_offset': -1.0, 'deg_rate': 0.08, 'max_laps': 15},
            'medium': {'lap_time_offset':  0.0, 'deg_rate': 0.05, 'max_laps': 35},
            'hard':   {'lap_time_offset':  1.5, 'deg_rate': 0.02, 'max_laps': 50},
        },
        'search_method': 'sampled',  # 'exhaustive' or 'sampled'
        'n_strategies_sampled': 500,    # only used if sampled
    }
    
    n_simulations = 1000
    optimal_laps = []
    optimal_laps_1_stop = []
    optimal_laps_2_stop_pit1 = []
    optimal_laps_2_stop_pit2 = []
    optimal_compounds_1_stop = []
    optimal_compounds_2_stop = []
      
    with Pool(processes=12) as pool:
        args = [(config, sim) for sim in range(n_simulations)]
        sim_results = []
        for i, result in enumerate(pool.imap(run_simulation, args)):
            sim_results.append(result)
            if i % (n_simulations // 10) == 0:
                print(f"Simulation {i}/{n_simulations}")
                sys.stdout.flush()
    
    for best_time_1, best_time_2, best_1, best_2, lap_times_1_stop, lap_times_2_stop in sim_results:
        pit1, c1 = best_1                    # e.g. pit1=20, c1=('soft','medium')
        pit2a, pit2b, c2 = best_2           # e.g. pit2a=20, pit2b=40, c2=('soft','medium','hard')
        
        best = min([(best_time_1, '1-stop', best_1),
                    (best_time_2, '2-stop', best_2)])
        
        optimal_laps.append(best[1])
        optimal_laps_1_stop.append(pit1)
        optimal_laps_2_stop_pit1.append(pit2a)
        optimal_laps_2_stop_pit2.append(pit2b)
        optimal_compounds_1_stop.append(c1)
        optimal_compounds_2_stop.append(c2)
        
    most_common_strategy = max(set(optimal_laps), key=optimal_laps.count)
    if most_common_strategy == '2-stop':
        plot_single_race(config, [pit2a, pit2b], list(c2))
    else:
        plot_single_race(config, [pit1], list(c1))

    # analyze distribution
    print(f"Most common optimal pit strategy: {max(set(optimal_laps), key=optimal_laps.count)}")
    print(f"Best 1-stop: pit lap {pit1}, compounds {c1}")
    print(f"Best 2-stop: pit laps {pit2a}/{pit2b}, compounds {c2}")
    
    compound_counts_1 = Counter(optimal_compounds_1_stop)
    top_combos = compound_counts_1.most_common(6)
    labels = [f"{c[0]}→{c[1]}" for c, _ in top_combos]
    values = [v for _, v in top_combos]

    plt.bar(labels, values)
    plt.xlabel("Compound Combination")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Compound Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_v7_compounds_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()
    
    compound_counts_2 = Counter(optimal_compounds_2_stop)
    top_combos = compound_counts_2.most_common(6)
    labels = [f"{c[0]}→{c[1]}→{c[2]}" for c, _ in top_combos]
    values = [v for _, v in top_combos]

    plt.bar(labels, values)
    plt.xlabel("Compound Combination")
    plt.ylabel("Frequency")
    plt.title(f"Best 2-Stop Compound Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_v7_compounds_2_stop.png', dpi=450, bbox_inches='tight')
    plt.close()
    
    plt.hist(optimal_laps_1_stop, bins=range(1, config['total_laps']), edgecolor='black')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps']/2), color='red', linestyle='--', label='Deterministic optimum')
    plt.legend()
    plt.savefig('monte_carlo_dist_v7_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()
    
    plt.hist(optimal_laps_2_stop_pit1, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='First stop')
    plt.hist(optimal_laps_2_stop_pit2, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='Second stop')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps']/3), color='red', linestyle='--', label='Deterministic optimum, stop 1')
    plt.axvline(x=(2*config['total_laps']/3), color='red', linestyle='--', label='Deterministic optimum, stop 2')
    plt.legend()
    plt.savefig('monte_carlo_dist_v7_2_stop.png', dpi=450, bbox_inches='tight')
    plt.close()
    
    counts = Counter(optimal_laps)
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best Pit Strategy Distribution ({n_simulations:,} simulations)")
    plt.savefig('monte_carlo_dist_v7_strategy.png', dpi=450, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()