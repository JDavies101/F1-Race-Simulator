# F1 Race Strategy Simulator

A Monte Carlo race strategy simulator for Formula 1, built in Python with an interactive Streamlit web interface. Given a set of race parameters, the simulator runs thousands of randomised race scenarios in parallel to produce a statistical distribution of optimal pit stop strategies — reflecting the uncertainty real strategists face on race day.

**[Live Demo →](https://f1-race-simulator.streamlit.app/)**

---
 
## What it does
 
Real F1 strategy is not deterministic — safety cars, traffic, and tire variability mean the "optimal" pit lap changes race to race. This simulator models that uncertainty by running N Monte Carlo simulations, each with randomised events, and asking: *across all of these scenarios, which strategy wins most often?*
 
Each simulation:
1. Generates a random safety car schedule for the race
2. Evaluates all candidate strategies (or a sampled subset) against that scenario
3. Records the best 1-stop and 2-stop strategy and which one wins overall
 
After N simulations, the output is a **distribution** of optimal strategies rather than a single answer — reflecting how real strategists think about race decisions under uncertainty.
 
---
 
## Features
 
- **Parallel simulation** via Python `ProcessPoolExecutor`, scaling across available CPU cores with a real-time progress bar in the web interface
- **Lap-by-lap physics model** with compound-specific degradation rates, triple-rate cliff behaviour past maximum tire life, and exponential traffic decay as the field spreads over a stint
- **Safety car model** with per-lap deployment probability and reduced pit delta under yellow flag conditions
- **1-stop and 2-stop strategy comparison** across all valid tire compound combinations, with the F1 regulation minimum 2-compound requirement enforced automatically
- **Exhaustive or sampled search** — evaluate every possible pit lap and compound combination, or randomly sample N strategies for faster interactive use
- **Track presets** for all 24 rounds of the 2026 F1 calendar with circuit-specific tire degradation, pit loss, and safety car parameters
- **Mobile layout toggle** for responsive viewing on any screen size
 
---
 
## Output
 
- **Strategy distribution** — how often 1-stop vs 2-stop wins across all simulations
- **Pit lap distributions** — where the optimal pit lap falls, coloured by opening compound to explain bimodal peaks
- **Compound frequency charts** — which tire combinations win most often for 1 and 2-stop strategies
- **Representative lap time traces** — single race trace for the most common 1-stop and 2-stop strategy, with stint shading by compound
- **Ranked strategy tables** — top 10 winning combinations for both 1 and 2-stop, with frequency and win rate calculated within each strategy subset
- **Summary metrics** — overall win rates, average best race times, and most common strategy
 
---
 
## Installation
 
```bash
git clone https://github.com/JDavies101/f1-strategy-simulator
cd f1-strategy-simulator
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## File Structure

```
streamlit_app.py          # Streamlit web interface
monte_carlo_medium_v1.py  # Core simulation engine (also runnable standalone)
track_presets.py          # Track configurations for all 24 2026 F1 races
requirements.txt          # Python dependencies
```

---

## Standalone Usage

The simulator can also be run directly without Streamlit:

```bash
python monte_carlo_medium_v1.py
```

This runs 1000 simulations with the default config and saves plots as PNG files. Edit the `config` dict in `main()` to change parameters.

---

## Key Parameters

| Parameter | Description |
|---|---|
| `base_lap_time` | Lap time on medium tires with no degradation (s) |
| `pit_loss` | Time lost in pit lane under green flag (s) |
| `sc_pit_loss` | Time lost in pit lane under safety car (s) |
| `sc_chance` | Safety car probability per race (%) |
| `deg_rate` | Tire degradation — seconds of lap time added per lap of tire age |
| `max_laps` | Tire age beyond which degradation triples (cliff behaviour) |
| `lap_time_offset` | Speed vs medium baseline (negative = faster) |
| `base_traffic_prob` | Probability of being in traffic at start of stint |
| `spread_rate` | Rate at which field spreads out over a stint (higher = slower decay) |
| `search_method` | `exhaustive` (optimal, slow) or `sampled` (fast, approximate) |

---

## Limitations & Future Work

- Single-car model — does not simulate multi-car interactions or undercut/overcut dynamics
- Tire parameters are approximations, not sourced from real telemetry data
- Safety car timing is randomly distributed per lap rather than event-triggered
- Potential future additions: virtual safety car, reliability failures, real tire compound data, multi-car simulation
 
---

## Author

Jake Davies — Aerospace Engineering MS, Auburn University  
Research focus: Rotating Detonation Rocket Engines, acoustic eigenfrequency analysis  
[LinkedIn](https://www.linkedin.com/in/jacob-davies/) · [GitHub](https://github.com/JDavies101)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
