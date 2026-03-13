# introduce multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool
import sys

def simulate_race(pit_lap, sc_laps, sc_base_lap_time, sc_pit_loss,
                  base_lap_time, pit_loss, tire_degradation_rate, total_laps,
                  pit_lap2=None):
    
    total_time = 0
    tire_age = 0
    lap_times = []
    
    pit_laps = [p for p in [pit_lap, pit_lap2] if p is not None]
    
    for laps in range(1, total_laps+1):
        
        pit_loss_randomness = np.random.normal(0, 2)
        tire_degradation_rate_randomness = np.random.normal(0, 0.005)
        
        if sc_laps[laps - 1]:
            current_base = sc_base_lap_time
        else:
            current_base = base_lap_time

        lap_time = current_base + (tire_degradation_rate + tire_degradation_rate_randomness) * tire_age

        if laps in pit_laps:
            if sc_laps[laps - 1]:
                lap_time += sc_pit_loss + pit_loss_randomness
            else:
                lap_time += pit_loss + pit_loss_randomness
            tire_age = 1
        else:
            tire_age += 1
        
        lap_times.append(lap_time)
        
        
    total_time = sum(lap_times)
    
    return total_time

def run_simulation(args):
    config, seed = args
    np.random.seed(seed)
    
    sc_laps = generate_safety_cars(config['total_laps'], config['sc_chance'])
    results_1_stop = {}
    results_2_stop = {}
    
    for pit_lap in range(1, config['total_laps']):
        results_1_stop[pit_lap] = simulate_race(
            pit_lap, sc_laps,
            config['sc_base_lap_time'], config['sc_pit_loss'],
            config['base_lap_time'], config['pit_loss'],
            config['tire_degradation_rate'], config['total_laps'],
            pit_lap2=None)
        
        for pit_lap2 in range(pit_lap + 1, config['total_laps']):
            results_2_stop[(pit_lap, pit_lap2)] = simulate_race(
                pit_lap, sc_laps,
                config['sc_base_lap_time'], config['sc_pit_loss'],
                config['base_lap_time'], config['pit_loss'],
                config['tire_degradation_rate'], config['total_laps'],
                pit_lap2=pit_lap2)
    
    best_1 = min(results_1_stop, key=results_1_stop.get)
    best_2 = min(results_2_stop, key=results_2_stop.get)
    
    return results_1_stop[best_1], results_2_stop[best_2], best_1, best_2

def generate_safety_cars(total_laps, sc_chance):
    
    return np.random.random(total_laps) < sc_chance

def main():
    
    config = {
        'base_lap_time': 81,
        'pit_loss': 25,
        'sc_pit_loss': 11,
        'sc_base_lap_time': 90,
        'tire_degradation_rate': 0.041,
        'total_laps': 60,
        'sc_chance': 50/60,
    }
    
    n_simulations = 10000
    optimal_laps = []
    optimal_laps_1_stop = []
    optimal_laps_2_stop_pit1 = []
    optimal_laps_2_stop_pit2 = []
    
    with Pool(processes=12) as pool:
        args = [(config, sim) for sim in range(n_simulations)]
        sim_results = []
        for i, result in enumerate(pool.imap(run_simulation, args)):
            sim_results.append(result)
            if i % (n_simulations // 10) == 0:
                print(f"Simulation {i}/{n_simulations}")
                sys.stdout.flush()
    
    for best_time_1, best_time_2, best_1, best_2 in sim_results:
        best = min([(best_time_1, '1-stop', best_1),
                    (best_time_2, '2-stop', best_2)])
        
        optimal_laps.append(best[1])
        optimal_laps_1_stop.append(best_1)
        optimal_laps_2_stop_pit1.append(best_2[0])
        optimal_laps_2_stop_pit2.append(best_2[1])

    # analyze distribution
    print(f"Most common optimal pit strategy: {max(set(optimal_laps), key=optimal_laps.count)}")
    print(f"Best 1-stop strategy: pit lap {best_1}")
    print(f"Best 2-stop strategy: pit lap {best_2[0]} and lap {best_2[1]}")
    
    plt.hist(optimal_laps_1_stop, bins=range(1, config['total_laps']), edgecolor='black')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 1-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps']/2), color='red', linestyle='--', label='Deterministic optimum')
    plt.legend()
    plt.savefig('monte_carlo_dist_v5_1_stop.png', dpi=450, bbox_inches='tight')
    plt.close()
    
    plt.hist(optimal_laps_2_stop_pit1, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='First stop')
    plt.hist(optimal_laps_2_stop_pit2, bins=range(1, config['total_laps']), edgecolor='black', alpha=0.7, label='Second stop')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best 2-Stop Pit Lap Distribution ({n_simulations:,} simulations)")
    plt.axvline(x=(config['total_laps']/3), color='red', linestyle='--', label='Deterministic optimum, stop 1')
    plt.axvline(x=(2*config['total_laps']/3), color='red', linestyle='--', label='Deterministic optimum, stop 2')
    plt.legend()
    plt.savefig('monte_carlo_dist_v5_2_stop.png', dpi=450, bbox_inches='tight')
    plt.close()
    
    counts = Counter(optimal_laps)
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title(f"Best Pit Strategy Distribution ({n_simulations:,} simulations)")
    plt.legend()
    plt.savefig('monte_carlo_dist_v5_strategy.png', dpi=450, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()