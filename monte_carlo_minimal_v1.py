# basic and deterministic simulation

import numpy as np

def simulate_race(pit_lap):
    
    base_lap_time = 81
    pit_loss = 20
    degradation_rate = 0.041
    total_laps = 60
    
    total_time = 0
    tire_age = 0
    lap_times = []
    
    for laps in range(1, total_laps+1):
        
        if laps == pit_lap:
            lap_time = base_lap_time + degradation_rate * tire_age + pit_loss
            tire_age = 1
        else:
            lap_time = base_lap_time + degradation_rate * tire_age
            tire_age += 1
        
        lap_times.append(lap_time)
        
        
    total_time = sum(lap_times)
    
    return total_time

def main():
    
    results = {}
    
    for pit_lap in range(1, 59):
        results[pit_lap] = simulate_race(pit_lap)
    
    optimal_lap = min(results, key=results.get)
    print(f"Optimal pit lap: ", optimal_lap)
    print(f"Best race time: {results[optimal_lap]:.2f}s")
    
if __name__ == "__main__":
    main()