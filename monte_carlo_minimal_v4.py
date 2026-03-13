# introduce 3 stop option - FAIL

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def simulate_race(pit_lap, sc_laps, sc_base_lap_time, sc_pit_loss, 
                  base_lap_time, pit_loss, tire_degradation_rate, total_laps,
                  pit_lap2 = None, pit_lap3 = None):
    
    total_time = 0
    tire_age = 0
    lap_times = []
    
    pit_laps = [p for p in [pit_lap, pit_lap2, pit_lap3] if p is not None]
    
    pit_loss_randomness = np.random.normal(0, 2)
    tire_degradation_rate_randomness = np.random.normal(0, 0.005)
    
    for laps in range(1, total_laps+1):
        
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

def generate_safety_cars(total_laps, sc_chance):
    
    return np.random.random(total_laps) < sc_chance

def main():
    
    n_simulations = 100
    optimal_laps = []
    sc_base_lap_time = 90
    sc_pit_loss = 11
    base_lap_time = 81
    pit_loss = 20
    tire_degradation_rate = 0.041
    total_laps = 60
    sc_general_chance = 50
    sc_chance = sc_general_chance / total_laps
    
    for sim in range(n_simulations):
        
        if sim % 10 == 0:
            print(f"Simulation {sim}")
    
        sc_laps = generate_safety_cars(total_laps, sc_chance)
        results_1_stop = {}
        results_2_stop = {}
        results_3_stop = {}
    
        # 1 stop simulation
        for pit_lap in range(1, 59):
            results_1_stop[(pit_lap)] = simulate_race(pit_lap,
                                sc_laps, sc_base_lap_time,
                                sc_pit_loss, base_lap_time, pit_loss, 
                                tire_degradation_rate, total_laps,
                                None, None)
            
            # 2 stop simulation
            for pit_lap2 in range(1, 59):
                results_2_stop[(pit_lap, pit_lap2)] = simulate_race(pit_lap,
                                    sc_laps, sc_base_lap_time,
                                    sc_pit_loss, base_lap_time, pit_loss, 
                                    tire_degradation_rate, total_laps, 
                                    pit_lap2, None)
            
                # 3 stop simulation
                for pit_lap3 in range(1, 59):
                    results_3_stop[(pit_lap, pit_lap2, pit_lap3)] = simulate_race(pit_lap,
                                        sc_laps, sc_base_lap_time,
                                        sc_pit_loss, base_lap_time, pit_loss, 
                                        tire_degradation_rate, total_laps, 
                                        pit_lap2, pit_lap3)
        
        best_1 = min(results_1_stop, key=results_1_stop.get)
        best_2 = min(results_2_stop, key=results_2_stop.get)
        best_3 = min(results_3_stop, key=results_3_stop.get)

        best_time_1 = results_1_stop[best_1]
        best_time_2 = results_2_stop[best_2]
        best_time_3 = results_3_stop[best_3]

        # find which strategy wins
        best = min([(best_time_1, '1-stop', best_1), 
                    (best_time_2, '2-stop', best_2), 
                    (best_time_3, '3-stop', best_3)])

        optimal_laps.append(best[1])  # record strategy type, not lap number

    # analyze distribution
    print(f"Most common optimal pit lap: {max(set(optimal_laps), key=optimal_laps.count)}")
    counts = Counter(optimal_laps)
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title("Distribution of Optimal Pit Lap (10,000 simulations)")
    plt.axvline(x=30, color='red', linestyle='--', label='Deterministic optimum')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()