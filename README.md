# F1 Race Strategy Simulator

A Monte Carlo pit strategy simulator for F1, written in Python with a Streamlit front end. It runs the same race thousands of times with random safety cars, traffic and tire wear, and shows which strategies win and how often.

[Live demo](https://f1-race-simulator.streamlit.app/)

## How it works

There's no single best pit lap in a real race because safety cars and traffic change things every time. So instead of solving for one answer, each simulation:

1. Generates random SC/VSC laps, traffic, pit stop times and a possible DNF
2. Runs every candidate strategy against that same race
3. Records the best 1-stop and 2-stop and which of the two was faster

After all the simulations you get a distribution of winning strategies rather than one answer.

## Model

- Lap by lap: base lap time, compound offset, and degradation that grows with tire age and triples past the compound's max laps
- Traffic is most likely right after a stop and drops off exponentially as the field spreads
- Safety car and virtual safety car slow the lap and make pitting cheaper
- Per-lap mechanical failure chance
- 1-stop and 2-stop strategies, with at least two different compounds as the rules require
- Two search modes:
  - exhaustive tries every pit lap and compound combination
  - sampled tries every 1-stop, samples 2-stops at random, then refines the best 2-stop for each compound set by checking nearby pit laps. It's much faster and usually within a second of exhaustive
- Presets for the 2026 calendar
- Seed input so runs can be repeated

## Output

- 1-stop vs 2-stop win counts
- Pit lap distributions, also split by starting compound (this explains the double peaks)
- Most common compound combinations
- A lap time trace for the top 1-stop and 2-stop
- Top 10 tables for each strategy type
- A zip download with the tables, parameters and lap traces

## Running it

```bash
git clone https://github.com/JDavies101/F1-Race-Simulator
cd F1-Race-Simulator
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The engine also runs on its own:

```bash
python monte_carlo.py
```

That runs the config in `main()` and saves the plots as PNGs.

Tests need pytest (`pip install pytest`), then from the repo root:

```bash
python -m pytest -v
```

## Files

```
streamlit_app.py   web app
monte_carlo.py     simulation engine, also runs standalone
track_presets.py   settings for each 2026 round
tests/             unit tests for the engine
requirements.txt
```

## Parameters

| Parameter | Description |
|---|---|
| `base_lap_time` | Lap time on fresh mediums (s) |
| `pit_loss` | Time lost pitting under green (s) |
| `sc_pit_loss`, `vsc_pit_loss` | Time lost pitting under SC / VSC (s) |
| `sc_chance`, `vsc_chance` | Chance of an SC / VSC per race (%) |
| `mechanical_failure_prob` | Chance of a failure per lap |
| `lap_time_offset` | Compound pace vs medium (negative is faster) |
| `deg_rate` | Seconds added per lap of tire age |
| `max_laps` | Tire age where deg triples |
| `base_traffic_prob` | Chance of traffic on the first lap of a stint |
| `spread_rate` | How slowly traffic clears (higher is slower) |
| `search_method` | `exhaustive` or `sampled` |

## Roadmap

Next:
- Fuel load, lap times dropping ~0.03s/lap as fuel burns off

Later:
- Weather that changes between dry, damp and wet during the race
- Track temperature tied to the weather
- Track rubbering in over the race

Long term:
- Separate degradation for each tire
- All 20 cars with their own strategies, for real traffic and undercuts
- Thermal tire model with graining and blistering

## Limitations

- Only one car, so no undercut/overcut or real traffic from other cars
- Tire numbers are estimates, not telemetry
- Safety cars happen at random laps rather than being caused by anything

## Author

Jake Davies, Aerospace Engineering MS, Auburn University  
[LinkedIn](https://www.linkedin.com/in/jacob-davies/) · [GitHub](https://github.com/JDavies101)

## License

MIT, see [LICENSE](LICENSE).
