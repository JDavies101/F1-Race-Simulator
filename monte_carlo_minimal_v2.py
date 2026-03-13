# introduce safety car variations
# introduce race simulation variability 
# introduce histogram

import numpy as np
import matplotlib.pyplot as plt

def simulate_race(pit_lap, sc_laps, sc_base_lap_time, sc_pit_loss, base_lap_time, pit_loss, degradation_rate, total_laps):
    
    total_time = 0
    tire_age = 0
    lap_times = []
    
    for laps in range(1, total_laps+1):
        
        if sc_laps[laps - 1]:
            current_base = sc_base_lap_time
        else:
            current_base = base_lap_time

        lap_time = current_base + degradation_rate * tire_age

        if laps == pit_lap:
            if sc_laps[laps - 1]:
                lap_time += sc_pit_loss
            else:
                lap_time += pit_loss
            tire_age = 1
        else:
            tire_age += 1
        
        lap_times.append(lap_time)
        
        
    total_time = sum(lap_times)
    
    return total_time

def generate_safety_cars(total_laps, sc_chance):
    
    return np.random.random(total_laps) < sc_chance

def main():
    
    n_simulations = 10000
    optimal_laps = []
    sc_chance = 0.008
    sc_base_lap_time = 90
    sc_pit_loss = 11
    base_lap_time = 81
    pit_loss = 20
    degradation_rate = 0.041
    total_laps = 60
    
    for sim in range(n_simulations):
        
        if sim % 1000 == 0:
            print(f"Simulation {sim}")
    
        sc_laps = generate_safety_cars(total_laps, sc_chance)
        results = {}
    
        for pit_lap in range(1, 59):
            results[pit_lap] = simulate_race(pit_lap, 
                                sc_laps, sc_base_lap_time,
                                sc_pit_loss, base_lap_time, pit_loss, 
                                degradation_rate, total_laps)
        
        optimal_lap = min(results, key=results.get)
        optimal_laps.append(optimal_lap)

    # analyze distribution
    print(f"Most common optimal pit lap: {max(set(optimal_laps), key=optimal_laps.count)}")
    plt.hist(optimal_laps, bins=range(1, 60), edgecolor='black')
    plt.xlabel("Optimal Pit Lap")
    plt.ylabel("Frequency")
    plt.title("Distribution of Optimal Pit Lap (10,000 simulations)")
    plt.axvline(x=30, color='red', linestyle='--', label='Deterministic optimum')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()