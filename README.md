# F1 Race Strategy Simulator

[![tests](https://github.com/JDavies101/F1-Race-Simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/JDavies101/F1-Race-Simulator/actions/workflows/tests.yml)

A Monte Carlo pit strategy simulator for F1, written in Python with a Streamlit front end. It runs the same race thousands of times with random safety cars, traffic and tire wear, and shows which strategies win and how often.

[Live demo](https://f1-race-simulator.streamlit.app/)

## How it works

There's no single best pit lap in a real race because safety cars and traffic change things every time. So instead of solving for one answer, each simulation:

1. Generates random SC/VSC laps, traffic, pit stop times and a possible DNF
2. Runs every candidate strategy against that same race
3. Records the best 1-stop and 2-stop and which of the two was faster

After all the simulations you get a distribution of winning strategies rather than one answer.

## Model

- Lap by lap: base lap time, compound offset, and degradation that grows with tire wear and triples past the compound's max laps
- Tires wear one lap's worth per lap, more on a heavy fuel load and in dirty air
- Fuel: the car gets faster as fuel burns off, and a full tank wears the tires faster
- Traffic is most likely right after a stop and drops off exponentially as the field spreads. Laps in traffic cost time and extra tire wear
- Safety car and virtual safety car slow the lap and make pitting cheaper
- Per-lap mechanical failure chance
- 1-stop and 2-stop strategies, with at least two different compounds as the rules require
- Two search modes:
  - exhaustive tries every pit lap and compound combination
  - sampled tries every 1-stop, samples 2-stops at random, then refines the best 2-stop for each compound set by checking nearby pit laps. It's much faster and usually within a second of exhaustive
- Presets for the 2026 calendar, fitted from real 2026 race data where the data allowed (see below)
- Seed input so runs can be repeated
- The simulation count defaults to what's needed for ±3% on the 1-stop/2-stop split, allowing for DNFs

## Output

- 1-stop vs 2-stop win counts, with the margin on the split at 95% confidence
- Pit lap distributions, also split by starting compound (this explains the double peaks), with the best stops for a race with no SC, traffic or noise marked for reference
- Most common compound combinations
- A lap time trace for the top 1-stop and 2-stop
- Top 10 tables for each strategy type
- A zip download with the tables, parameters and lap traces

## Calibrated from real races

`calibrate.py` fits the presets from 2026 race laps using [FastF1](https://github.com/theOehrly/Fast-F1). It keeps clean green-flag laps on dry tires, skips races with too few of them, and fits:

- **Compound pace** by Pirelli compound (C1 to C5) across all races at once, since a "soft" is a different tire from track to track
- **Degradation and base lap time** for each race on its own, since wear depends on the track as much as the tire
- **Fuel effect**, one value shared by every race

A fitted value is only used if it comes from enough laps, has the right sign and is clear of the noise. Everything else keeps the hand-set estimate in `track_presets.py`. The fitted values go to `calibrated_presets.json`, which `track_presets.py` applies when it loads.

What the 2026 data showed:

- Fuel costs about 0.035s per lap of fuel on board
- Neighbouring compounds are closer than the usual estimates: C3 and C4 are about the same pace, C2 is about 0.5s slower than C3 and C5 about 0.15s faster than C4
- Race to race, compound gaps vary by about 0.3s, more than any single race's fit suggests
- Soft degradation never passed the checks. Softs mostly run short early stints on used tires, which makes their wear hard to separate from the race phase

Running it needs an F1 TV Access account, since F1 put its timing archive behind a login in October 2026:

```bash
pip install -r requirements-dev.txt
python -m fastf1 auth f1tv --authenticate
python calibrate.py
```

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
streamlit_app.py          web app
monte_carlo.py            simulation engine, also runs standalone
track_presets.py          settings for each 2026 round
calibrate.py              fits presets from real race data, not used by the app
calibrated_presets.json   fitted values applied over track_presets.py
tests/                    unit tests
requirements.txt
requirements-dev.txt      extra packages for calibrate.py
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
| `max_laps` | Laps of wear where deg triples |
| `fuel_effect` | Seconds per lap slower for each lap of fuel on board |
| `fuel_wear` | Extra tire wear per lap on a full tank, fading to 0 by the end |
| `dirty_air_wear` | Extra tire wear per lap in traffic |
| `base_traffic_prob` | Chance of traffic on the first lap of a stint |
| `spread_rate` | How slowly traffic clears (higher is slower) |
| `search_method` | `exhaustive` or `sampled` |

## Roadmap

Next:
- Weather that changes between dry, damp and wet during the race
- Track temperature tied to the weather
- Soft compound data from practice long runs, where teams run fresh softs over many laps

Long term:
- Separate degradation for each tire
- All 20 cars with their own strategies, for real traffic and undercuts
- Thermal tire model with graining and blistering

## Limitations

- Only one car, so no undercut/overcut or real traffic from other cars
- Presets are fitted from one season of race laps. Soft degradation, tire cliffs and tracks without clean races still use estimates
- Compound pace is an average across races and won't match any single race
- Safety cars happen at random laps rather than being caused by anything

## Author

Jake Davies, Aerospace Engineering MS, Auburn University  
[LinkedIn](https://www.linkedin.com/in/jacob-davies/) · [GitHub](https://github.com/JDavies101)

## License

MIT, see [LICENSE](LICENSE).
