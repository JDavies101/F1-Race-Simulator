# Fits tire and fuel numbers from real race laps for track_presets.py.
# Needs fastf1 3.9+ and an F1 TV Access login (python -m fastf1 auth f1tv --authenticate).
# Not used by the app.

import os
import fastf1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

os.makedirs('fastf1_cache', exist_ok=True) # fastf1 wont create it
fastf1.Cache.configure(cache_dir='fastf1_cache')
fastf1.set_log_level('ERROR')   # hide fastf1's per-race info and warnings

COMPOUNDS = ['soft', 'medium', 'hard']

# Pirelli's compound choice for each 2026 race, from their race previews
ALLOCATION = {
    'Australia': {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'China':     {'hard': 'C2', 'medium': 'C3', 'soft': 'C4'},
    'Japan':     {'hard': 'C1', 'medium': 'C2', 'soft': 'C3'},
    'Miami':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'Canada':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'Monaco':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'Barcelona':     {'hard': 'C2', 'medium': 'C3', 'soft': 'C4'},
    'Austria':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'British':     {'hard': 'C1', 'medium': 'C2', 'soft': 'C3'},
    'Belgium':     {'hard': 'C2', 'medium': 'C3', 'soft': 'C4'},
    'Hungary':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'Dutch':     {'hard': 'C2', 'medium': 'C3', 'soft': 'C4'},
    'Italy':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'Spain':     {'hard': 'C2', 'medium': 'C3', 'soft': 'C4'},
    'Azerbaijan':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
    'Bahrain':     {'hard': 'C2', 'medium': 'C3', 'soft': 'C4'},
    'Singapore':     {'hard': 'C3', 'medium': 'C4', 'soft': 'C5'},
}

# calibrate.py event key -> preset name in track_presets.py (None = no preset yet)
PRESET_NAME = {
    'Australia': 'Australia (Melbourne)', 'China': 'China (Shanghai)', 'Japan': 'Japan (Suzuka)',
    'Miami': 'Miami', 'Canada': 'Canada (Montreal)', 'Monaco': 'Monaco',
    'Barcelona': 'Barcelona-Catalunya', 'Austria': 'Austria (Spielberg)',
    'British': 'Great Britain (Silverstone)', 'Belgium': 'Belgium (Spa)',
    'Hungary': 'Hungary (Budapest)', 'Dutch': 'Netherlands (Zandvoort)', 'Italy': 'Italy (Monza)',
    'Spain': 'Spain (Madrid)', 'Azerbaijan': 'Azerbaijan (Baku)', 'Bahrain': 'Bahrain (Sakhir)',
    'Singapore': 'Singapore',
}

# (race, compound) degradation known to be confounded, checked by hand
SKIP_DEG = {('Belgian Grand Prix', 'medium')}


def preset_updates(races, fit, err, fuel, min_sigma=2.5):
    """
    Values to apply over track_presets.py, per preset name. Offsets come from
    the pooled C number fit, deg and base lap time from each race's own fit.
    Anything that fails a check is left out so the hand-set value stays.
    """
    off = {c: fit.get(('offset', c), 0.0) for c in ['C1', 'C2', 'C3', 'C4', 'C5']}
    se = {c: err.get(('offset', c), 0.0) for c in off}
    deg_key = {'soft': 'soft_deg', 'medium': 'med_deg', 'hard': 'hard_deg'}

    updates = {}
    for laps in races:
        race, key = laps['Race'].iloc[0], laps['Key'].iloc[0]
        preset = PRESET_NAME.get(key)
        if preset is None:
            continue
        alloc = ALLOCATION[key]
        values = {'fuel_effect': round(fuel, 3)}

        # compound gaps vs this track's medium. too close to separate, or the
        # wrong way round, -> 0
        for comp, name, sign in [('soft', 'soft_offset', -1), ('hard', 'hard_offset', +1)]:
            gap = off[alloc[comp]] - off[alloc['medium']]
            gap_se = (se[alloc[comp]] ** 2 + se[alloc['medium']] ** 2) ** 0.5
            clear = abs(gap) > min_sigma * gap_se and gap * sign > 0
            values[name] = round(gap, 2) if clear else 0.0

        try:
            r = fit_race(laps, int(laps['TotalLaps'].iloc[0]), fuel=fuel)
        except ValueError:
            updates[preset] = values
            continue
        values['base_lap_time'] = round(r['base_lap_time'], 1)
        for comp, v in accepted_values(r, min_sigma=min_sigma).items():
            if v['deg'] is not None and (race, comp) not in SKIP_DEG:
                values[deg_key[comp]] = round(v['deg'], 3)

        updates[preset] = values
    return updates

def fit_pooled(laps, fuel=None, reference='C3'):
    """
    One fit over many races at once:
        lap_time = pace[race, driver] + offset[C number] + deg[race, compound] * tire_age
                   + track[race] * exp(-lap / 10) + fuel * laps_remaining
                   + stint_effect[stint position]
    offset is shared across races by Pirelli compound, with `reference` at 0.
    stint_effect is shared too: the first stint is the reference, then one
    term for the second and one for any later stint. It keeps first-stint
    traffic and tire saving from being blamed on whichever compound usually
    starts.
    deg stays per race, since wear depends on the track as much as the tire.
    fuel: s per lap of fuel, fixed when given, otherwise fitted as one value
    shared by every race.
    """
    laps = laps.reset_index(drop=True)
    tire_age = laps['TyreLife'].to_numpy() - 1
    laps_left = (laps['TotalLaps'] - laps['LapNumber']).to_numpy().astype(float)
    y = laps['LapTime'].to_numpy()
    if fuel is not None:
        y = y - fuel * laps_left

    columns = []
    names = []

    for (race, driver), rows in laps.groupby(['Race', 'Driver']).groups.items():
        col = np.zeros(len(laps))
        col[rows] = 1
        columns.append(col)
        names.append(('pace', race, driver))

    for cnum in sorted(laps['CNum'].unique()):
        if cnum != reference:
            columns.append((laps['CNum'] == cnum).to_numpy(float))
            names.append(('offset', cnum))

    for (race, comp), rows in laps.groupby(['Race', 'Compound']).groups.items():
        col = np.zeros(len(laps))
        col[rows] = tire_age[rows]
        columns.append(col)
        names.append(('deg', race, comp))

    for race in laps['Race'].unique():
        is_race = (laps['Race'] == race).to_numpy(float)
        columns.append(is_race * np.exp(-laps['LapNumber'].to_numpy() / 10))
        names.append(('track', race))

    # Stint counts tire changes from the start, so 1 is the first stint
    stint = laps['Stint'].to_numpy()
    columns.append((stint == 2).astype(float))
    names.append(('stint', 2))
    columns.append((stint >= 3).astype(float))
    names.append(('stint', 3))

    if fuel is None:
        columns.append(laps_left)
        names.append(('fuel',))

    x = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef

    dof = len(y) - x.shape[1]
    sigma2 = residual @ residual / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(x.T @ x)))
    return dict(zip(names, coef)), dict(zip(names, se)), float(np.sqrt(np.mean(residual ** 2)))

def accepted_values(result, min_laps=150, min_sigma=2.5):
    """
    Keep a compound's fitted numbers only where the data supports them.
    Returns {compound: {'offset': ..., 'deg': ...}} with None for rejected
    values, so the caller keeps the hand-set preset there.
    """
    # a significantly negative deg means this race's fit is confounded, so
    # don't trust anything from it
    if any(result['deg'][c] < -min_sigma * result['deg_se'][c] for c in result['deg']):
        return {c: {'offset': None, 'deg': None} for c in result['deg']}

    out = {}

    for c in result['deg']:
        enough = result['laps_used'][c] >= min_laps
        deg, deg_se = result['deg'][c], result['deg_se'][c]
        off, off_se = result['offset'][c], result['offset_se'][c]

        deg_ok = enough and deg >= 0 and deg > min_sigma * deg_se
        # medium is the reference so its offset is always 0. other offsets
        # must be clear of the noise and on the right side of the medium
        if c == 'medium':
            off_ok = True
        else:
            right_side = off < 0 if c == 'soft' else off > 0
            off_ok = enough and right_side and abs(off) > min_sigma * off_se

        out[c] = {'offset': off if off_ok else None, 'deg': deg if deg_ok else None}

    return out

def plot_fit(laps, total_laps, fit, pace):
    """Lap time minus driver pace and fuel, against tire age, one color per compound."""

    tire_age = laps['TyreLife'] - 1
    adjusted = (laps['LapTime'] - laps['Driver'].map(pace)
                - fit['fuel_effect'] * (total_laps - laps['LapNumber'])
                - fit['track_evolution'] * np.exp(-laps['LapNumber'] / 10))
    
    colors = {'soft': 'red', 'medium': 'gold', 'hard': 'gray'}

    for c, color in colors.items():
        mask = laps['Compound'] == c
        if not mask.any():
            continue
        plt.scatter(tire_age[mask], adjusted[mask], s=8, color=color, alpha=0.5, label=c)
        age = np.arange(0, tire_age[mask].max() + 1)
        plt.plot(age, fit['offset'][c] + fit['deg'][c] * age, color=color)

    plt.xlabel('Tire age (laps)')
    plt.ylabel('Lap time minus driver pace and fuel (s)')
    plt.legend()
    plt.savefig('fit_check.png', dpi=150, bbox_inches='tight')
    plt.close()

def load_race_laps(year, event):
    """Clean green flag racing laps on dry tires from one race"""

    session = fastf1.get_session(year, event, 'R')
    session.load(telemetry=False, weather=False, messages=False)

    laps = (session.laps
            .pick_wo_box() # no in or out laps
            .pick_track_status('1') # green flag the whole lap
            .pick_accurate()
            .pick_compounds(['soft', 'medium', 'hard'])
            .pick_quicklaps(1.07)) # drop laps over 107% of the fastest

    laps = laps[laps['LapNumber'] > 1]

    out = laps[['Driver', 'LapNumber', 'Compound', 'TyreLife', 'Stint']].copy()
    out['LapTime'] = laps['LapTime'].dt.total_seconds()
    out['Compound'] = out['Compound'].str.lower()
    out = out.dropna()
    stint_len = out.groupby(['Driver', 'Stint'])['LapNumber'].transform('count')
    out = out[stint_len >= 5].copy()
    name = session.event['EventName']
    out['Race'] = name
    out['Key'] = event
    out['TotalLaps'] = int(session.total_laps)
    out['CNum'] = out['Compound'].map(ALLOCATION[event])

    return out, int(session.total_laps)

def fit_race(laps, total_laps, fuel=None):
    """
    Least squares fit of
        lap_time = driver_pace + offset[compound] + deg[compound] * tire_age
                   + fuel_effect * laps_remaining
    with medium as the reference (offset 0), like the simulator.
    A pace term per driver keeps car speed out of the compound numbers.
    """
    drivers = sorted(laps['Driver'].unique())
    compounds = [c for c in COMPOUNDS if c in set(laps['Compound'])]

    if 'medium' not in compounds:
        raise ValueError("no medium laps, offsets need medium as the reference")

    tire_age = laps['TyreLife'].to_numpy() - 1 # simulator counts from 0
    laps_left = total_laps - laps['LapNumber'].to_numpy()

    columns = []
    names = []
    for d in drivers:
        columns.append((laps['Driver'] == d).to_numpy(float))
        names.append(('pace', d))

    for c in compounds:
        is_c = (laps['Compound'] == c).to_numpy(float)
        if c!= 'medium':
            columns.append(is_c)
            names.append(('offset', c))

        columns.append(is_c * tire_age)
        names.append(('deg', c))
    
    # track rubbering in: fast early gains that level off. not a straight
    # line, so it can be separated from fuel
    columns.append(np.exp(-laps['LapNumber'].to_numpy() / 10))
    names.append(('track', None))

    y = laps['LapTime'].to_numpy()

    if fuel is None:
        columns.append(laps_left.astype(float))
        names.append(('fuel', None))
    else:
        y = y - fuel * laps_left          # take the known fuel effect out first

    x = np.column_stack(columns)

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    fit = dict(zip(names, coef))
    residual = y - x @ coef

    # standard error of each coefficient: how far it could be off from noise alone
    dof = len(y) - x.shape[1]
    sigma2 = residual @ residual / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(x.T @ x)))
    err = dict(zip(names, se))

    return {
        # empty tank, fresh mediums, a typical driver: same meaning as the simulator's base
        'base_lap_time': float(np.median([fit[('pace', d)] for d in drivers])),
        'offset': {c: float(fit.get(('offset', c), 0.0)) for c in compounds},
        'deg': {c: float(fit[('deg', c)]) for c in compounds},
        'laps_used': {c: int((laps['Compound'] == c).sum()) for c in compounds},
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'pace': {d: float(fit[('pace', d)]) for d in drivers},
        'fuel_effect': float(fit.get(('fuel', None), fuel)),
        'track_evolution': float(fit[('track', None)]),
        'offset_se': {c: float(err.get(('offset', c), 0.0)) for c in compounds},
        'deg_se': {c: float(err[('deg', c)]) for c in compounds},
    }

def max_stint_laps(laps, pct=90):
    """Long end of the stints teams actually ran, a lower bound for the cliff."""

    longest = laps.groupby(['Driver', 'Stint', 'Compound'])['TyreLife'].max().reset_index()
    return {c: int(np.percentile(g['TyreLife'], pct)) for c, g in longest.groupby('Compound')}


if __name__ == '__main__':

    MIN_CLEAN = 0.5   # skip races where too few laps were green flag racing

    races, summary = [], {}
    for event in ALLOCATION:
        try:
            laps, total_laps = load_race_laps(2026, event)
        except Exception as e:   # not run yet, failed download, etc.
            print(f"{event:20} skipped: {e}")
            continue

        name = laps['Race'].iloc[0]
        clean = len(laps) / (total_laps * laps['Driver'].nunique())
        if name in summary:
            print(f"{event:20} skipped: same race as an earlier key ({name})")
            continue
        summary[name] = {'key': event, 'clean_fraction': round(clean, 2),
                         'laps': {c: int(n) for c, n in laps['CNum'].value_counts().items()}}
        if clean < MIN_CLEAN:
            print(f"{event:20} skipped: only {clean:.0%} clean ({name})")
            continue

        print(f"{event:20} {name:30} {clean:.0%} clean, {len(laps)} laps")
        races.append(laps)

    # races whose compound use is too tied to one phase of the race to measure
    # compound pace. checked by hand from the stint table
    NO_OFFSETS = {'Belgian Grand Prix'}
    offset_laps = pd.concat([l for l in races if l['Race'].iloc[0] not in NO_OFFSETS])

    # how the compound order responds to the fuel value
    print("\nfuel   rmse    offsets vs C3")
    for f in [0.055, 0.07, 0.085, 0.10]:
        fit, err, rmse = fit_pooled(offset_laps, fuel=f)
        order = '  '.join(f"{k[1]} {fit[k]:+.2f}" for k in sorted(fit) if k[0] == 'offset')
        print(f"{f:.3f}  {rmse:.4f}  {order}")

    fit, err, rmse = fit_pooled(offset_laps)   # fuel fitted from the data
    fuel = fit[('fuel',)]
    print(f"\nfitted fuel {fuel:.4f} +- {err[('fuel',)]:.4f}")
    print(f"stint 2 {fit[('stint', 2)]:+.3f} +- {err[('stint', 2)]:.3f}, "
          f"stint 3+ {fit[('stint', 3)]:+.3f} +- {err[('stint', 3)]:.3f} (vs first stint)")

    offsets = {k[1]: {'value': round(fit[k], 3), 'se': round(err[k], 3)}
               for k in sorted(fit) if k[0] == 'offset'}
    deg = {f"{k[1]} {k[2]}": {'value': round(fit[k], 4), 'se': round(err[k], 4)}
           for k in sorted(fit) if k[0] == 'deg'}

    print(f"\n{offset_laps['Race'].nunique()} races, rmse {rmse:.3f}")
    for c, v in offsets.items():
        print(f"{c} {v['value']:+.3f} +- {v['se']:.3f}")

    print("\nper race vs pooled (soft-medium, hard-medium)")
    pooled = {c: fit.get(('offset', c), 0.0) for c in ['C1', 'C2', 'C3', 'C4', 'C5']}
    for laps in races:
        name = laps['Race'].iloc[0]
        alloc = laps.groupby('Compound')['CNum'].first().to_dict()
        try:
            r = fit_race(laps, int(laps['TotalLaps'].iloc[0]), fuel=round(fuel, 3))
        except ValueError:
            continue
        line = []
        for comp in ['soft', 'hard']:
            if comp in r['offset'] and 'medium' in alloc:
                expect = pooled[alloc[comp]] - pooled[alloc['medium']]
                got, se = r['offset'][comp], r['offset_se'][comp]
                flag = ' <--' if abs(got - expect) > 2.5 * se else ''
                line.append(f"{comp} {got:+.2f}+-{se:.2f} (pooled {expect:+.2f}){flag}")
        print(f"{name:28} {alloc.get('medium')}  " + '   '.join(line))

    with open('calibration_results.json', 'w') as f:
        json.dump({'fuel': round(fuel, 4), 'fuel_se': round(err[('fuel',)], 4), 'rmse': round(rmse, 3), 'races': summary,
                   'offsets': offsets, 'deg': deg}, f, indent=2)
    print("details in calibration_results.json")

    updates = preset_updates(races, fit, err, fuel)
    with open('calibrated_presets.json', 'w') as f:
        json.dump(updates, f, indent=2, sort_keys=True)
    print(f"wrote fitted values for {len(updates)} presets to calibrated_presets.json")
