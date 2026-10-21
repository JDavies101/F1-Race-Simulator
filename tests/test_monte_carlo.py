# Tests for monte_carlo.py. Run from the repo root with: python -m pytest -v
#
# Most tests use a short 6-lap race with no randomness so lap times can be
# worked out by hand. Numbers in the comments show that working.

import numpy as np
import pytest

from monte_carlo import (calm_optimum, compound_label, dnf_rate, generate_race_events, race_totals,
                         run_simulation, simulate_race, sims_needed, summarize, win_share_margin)

LAPS = 6
GREEN = np.zeros(LAPS, dtype=int)


def small_config(**overrides):
    config = {
        'total_laps': LAPS,
        'base_lap_time': 80, 'sc_base_lap_time': 90, 'vsc_base_lap_time': 85,
        'pit_loss': 20, 'sc_pit_loss': 8, 'vsc_pit_loss': 14,
        'base_traffic_prob': 0.0, 'spread_rate': 10, 'traffic_time_loss': 1.0,
        'compound_data': {
            'soft':   {'lap_time_offset': -1.0, 'deg_rate': 0.10, 'max_laps': 2},
            'medium': {'lap_time_offset':  0.0, 'deg_rate': 0.05, 'max_laps': 10},
            'hard':   {'lap_time_offset':  1.0, 'deg_rate': 0.02, 'max_laps': 10},
        },
    }
    config.update(overrides)
    return config


def no_noise(pit=0.0, deg=0.0, traffic=1.0):
    # traffic draw of 1.0 is never below a probability, so no traffic
    return {'pit': np.full(LAPS, pit), 'deg': np.full(LAPS, deg), 'traffic': np.full(LAPS, traffic)}


# generate_race_events

def test_events_all_sc_when_sc_chance_is_one():
    events = generate_race_events(10, 1.0, 1.0, np.random.default_rng(0))
    assert events.tolist() == [2] * 10


def test_events_all_vsc_when_only_vsc_chance_is_one():
    events = generate_race_events(10, 0.0, 1.0, np.random.default_rng(0))
    assert events.tolist() == [1] * 10


def test_events_all_green_when_no_chance():
    events = generate_race_events(10, 0.0, 0.0, np.random.default_rng(0))
    assert events.tolist() == [0] * 10


def test_events_same_seed_gives_same_events():
    a = generate_race_events(200, 0.1, 0.1, np.random.default_rng(42))
    b = generate_race_events(200, 0.1, 0.1, np.random.default_rng(42))
    assert a.tolist() == b.tolist()
    assert set(a.tolist()) == {0, 1, 2}   # all three event types show up over 200 laps


# simulate_race lap times

def test_one_stop_lap_times():
    # medium laps 1-3 (age 0,1,2), pit on 3, hard laps 4-6 (age 1,2,3)
    result = simulate_race(3, ('medium', 'hard'), GREEN, small_config(), noise=no_noise())
    assert result['lap_times'] == pytest.approx([80.0, 80.05, 80.10 + 20, 81.02, 81.04, 81.06])
    assert result['total_time'] == pytest.approx(503.27)
    assert result['dnf'] is False


def test_two_stop_lap_times():
    # soft 1-2, pit on 2, medium 3-4, pit on 4, hard 5-6
    result = simulate_race(2, ('soft', 'medium', 'hard'), GREEN, small_config(),
                           pit_lap2=4, noise=no_noise())
    assert result['lap_times'] == pytest.approx([79.0, 79.1 + 20, 80.05, 80.10 + 20, 81.02, 81.04])


def test_tire_cliff_triples_deg_past_max_laps():
    # soft max_laps is 2: age 3 and 4 use 0.3 deg instead of 0.1
    result = simulate_race(5, ('soft', 'medium'), GREEN, small_config(), noise=no_noise())
    assert result['lap_times'] == pytest.approx([79.0, 79.1, 79.2, 79.9, 80.2 + 20, 80.05])


@pytest.mark.parametrize('event, expected_pit_lap', [
    (0, 80 + 0.10 + 20),   # green: base 80, full pit loss
    (1, 85 + 0.10 + 14),   # VSC: base 85, vsc pit loss
    (2, 90 + 0.10 + 8),    # SC: base 90, sc pit loss
])
def test_pit_cost_depends_on_event(event, expected_pit_lap):
    events = GREEN.copy()
    events[2] = event
    result = simulate_race(3, ('medium', 'hard'), events, small_config(), noise=no_noise())
    assert result['lap_times'][2] == pytest.approx(expected_pit_lap)


def test_pit_noise_only_added_on_pit_lap():
    result = simulate_race(3, ('medium', 'hard'), GREEN, small_config(), noise=no_noise(pit=1.5))
    assert result['lap_times'] == pytest.approx([80.0, 80.05, 80.10 + 20 + 1.5, 81.02, 81.04, 81.06])


def test_deg_noise_scales_with_tire_age():
    # +0.01 deg per lap of age: ages are 0,1,2,1,2,3
    result = simulate_race(3, ('medium', 'hard'), GREEN, small_config(), noise=no_noise(deg=0.01))
    assert result['lap_times'] == pytest.approx([80.0, 80.06, 80.12 + 20, 81.03, 81.06, 81.09])


def test_traffic_fades_over_stint_and_resets_after_stop():
    # prob = 0.5 * exp(-stint_lap / 10): 0.5, 0.452, 0.409 for stint laps 0, 1, 2.
    # draw of 0.45 means traffic on stint laps 0 and 1 only. stint laps are
    # 0,1,2 then 0,1,2 after the stop on lap 3, so traffic on laps 1, 2, 4, 5
    config = small_config(base_traffic_prob=0.5)
    result = simulate_race(3, ('medium', 'hard'), GREEN, config, noise=no_noise(traffic=0.45))
    assert result['lap_times'] == pytest.approx([81.0, 81.05, 80.10 + 20, 82.02, 82.04, 81.06])


def test_no_traffic_loss_under_safety_car():
    events = GREEN.copy()
    events[0] = 2
    config = small_config(base_traffic_prob=0.5)
    result = simulate_race(3, ('medium', 'hard'), events, config, noise=no_noise(traffic=0.45))
    assert result['lap_times'][0] == pytest.approx(90.0)   # SC lap, no +1.0 traffic
    assert result['lap_times'][1] == pytest.approx(81.05)  # green lap, still in traffic


def test_fuel_effect_slows_heavy_laps():
    # 0.1s per lap of fuel left after this lap: +0.5 on lap 1 down to 0 on lap 6
    config = small_config(fuel_effect=0.1)
    result = simulate_race(3, ('medium', 'hard'), GREEN, config, noise=no_noise())
    assert result['lap_times'] == pytest.approx([80.5, 80.45, 100.40, 81.22, 81.14, 81.06])


def test_fuel_effect_not_applied_under_safety_car():
    events = GREEN.copy()
    events[0] = 2
    config = small_config(fuel_effect=0.1)
    result = simulate_race(3, ('medium', 'hard'), events, config, noise=no_noise())
    assert result['lap_times'][0] == pytest.approx(90.0)
    assert result['lap_times'][1] == pytest.approx(80.45)


def test_fuel_wear_adds_tire_wear_on_full_tank():
    # fuel left (6 - lap) / 5 is 1, .8, .6, .4, .2, 0, so wear per lap is
    # 1.5, 1.4, 1.3, 1.2, 1.1, 1.0. wear before each lap: 0, 1.5, 2.9, then
    # after the stop on lap 3: 1, 2.2, 3.3
    config = small_config(fuel_wear=0.5)
    result = simulate_race(3, ('medium', 'hard'), GREEN, config, noise=no_noise())
    assert result['lap_times'] == pytest.approx([80.0, 80.075, 80.145 + 20, 81.02, 81.044, 81.066])


def test_dirty_air_adds_tire_wear_in_traffic():
    # traffic on laps 1, 2, 4, 5 (see the traffic test above), so wear per lap
    # is 2, 2, 1, 2, 2, 1 and wear before each lap is 0, 2, 4, then 1, 3, 5
    config = small_config(base_traffic_prob=0.5, dirty_air_wear=1.0)
    result = simulate_race(3, ('medium', 'hard'), GREEN, config, noise=no_noise(traffic=0.45))
    assert result['lap_times'] == pytest.approx([81.0, 81.10, 80.20 + 20, 82.02, 82.06, 81.10])


def test_cliff_uses_wear_not_laps():
    # soft max_laps is 2. lap 3 is only 2 laps old, but with fuel wear it has
    # 2.9 laps of wear, so it's past the cliff: 79 + 0.3 * 2.9
    config = small_config(fuel_wear=0.5)
    result = simulate_race(5, ('soft', 'medium'), GREEN, config, noise=no_noise())
    assert result['lap_times'][2] == pytest.approx(79.87)


def test_dnf_returns_inf_and_no_laps():
    result = simulate_race(3, ('medium', 'hard'), GREEN, small_config(), dnf=True, dnf_lap=4)
    assert result == {'total_time': np.inf, 'lap_times': [], 'dnf': True, 'dnf_lap': 4}


def test_simulate_race_makes_noise_from_rng_when_none_given():
    a = simulate_race(3, ('medium', 'hard'), GREEN, small_config(), rng=np.random.default_rng(7))
    b = simulate_race(3, ('medium', 'hard'), GREEN, small_config(), rng=np.random.default_rng(7))
    assert a['lap_times'] == b['lap_times']
    assert a['lap_times'] != pytest.approx([80.0, 80.05, 100.10, 81.02, 81.04, 81.06])  # noise was applied


# race_totals

def test_race_totals_matches_hand_totals_for_several_strategies():
    strategies = [(3, None, ('medium', 'hard')), (2, 4, ('soft', 'medium', 'hard'))]
    totals = race_totals(strategies, GREEN, small_config(), no_noise())
    # second total: 79 + 99.1 + 80.05 + 100.10 + 81.02 + 81.04
    assert totals.tolist() == pytest.approx([503.27, 520.31])


def test_race_totals_all_inf_on_dnf():
    strategies = [(3, None, ('medium', 'hard')), (2, 4, ('soft', 'medium', 'hard'))]
    totals = race_totals(strategies, GREEN, small_config(), no_noise(), dnf=True)
    assert totals.tolist() == [np.inf, np.inf]


# run_simulation

def sim_config(**overrides):
    config = {
        'base_lap_time': 81, 'pit_loss': 25, 'sc_pit_loss': 12, 'sc_base_lap_time': 90,
        'vsc_pit_loss': 18, 'vsc_base_lap_time': 87, 'mechanical_failure_prob': 0.0,
        'total_laps': 30, 'sc_chance': 0.02, 'vsc_chance': 0.02,
        'base_traffic_prob': 0.4, 'spread_rate': 15, 'traffic_time_loss': 0.5,
        'compound_data': {
            'soft':   {'lap_time_offset': -0.6, 'deg_rate': 0.08, 'max_laps': 15},
            'medium': {'lap_time_offset':  0.0, 'deg_rate': 0.05, 'max_laps': 35},
            'hard':   {'lap_time_offset':  0.6, 'deg_rate': 0.02, 'max_laps': 50},
        },
        'search_method': 'sampled', 'n_strategies_sampled': 100,
    }
    config.update(overrides)
    return config


def test_run_simulation_same_seed_same_result():
    assert run_simulation((sim_config(), 3)) == run_simulation((sim_config(), 3))


def test_run_simulation_different_seeds_differ():
    assert run_simulation((sim_config(), 3))[:2] != run_simulation((sim_config(), 4))[:2]


@pytest.mark.parametrize('method', ['sampled', 'exhaustive'])
def test_run_simulation_strategies_follow_rules(method):
    best_time_1, best_time_2, best_1, best_2, dnf_count = run_simulation((sim_config(search_method=method), 0))
    pit, comps_1 = best_1
    pit1, pit2, comps_2 = best_2
    assert comps_1[0] != comps_1[1]          # two different compounds
    assert len(set(comps_2)) >= 2
    assert 1 <= pit < 30
    assert 1 <= pit1 < pit2 < 30
    assert dnf_count == 0
    assert np.isfinite(best_time_1) and np.isfinite(best_time_2)


def test_run_simulation_both_modes_find_same_best_one_stop():
    # every 1-stop is tried in both modes, so they must agree
    sampled = run_simulation((sim_config(search_method='sampled'), 5))
    exhaustive = run_simulation((sim_config(search_method='exhaustive'), 5))
    assert sampled[2] == exhaustive[2]
    assert sampled[0] == pytest.approx(exhaustive[0])


@pytest.mark.parametrize('laps', [10, 15, 20, 21])
@pytest.mark.parametrize('seed', range(5))
def test_sampled_short_races_give_valid_strategies(laps, seed):
    # under 21 laps the normal sampling window (lap 5 to laps-15) is empty
    _, best_time_2, best_1, best_2, _ = run_simulation((sim_config(total_laps=laps), seed))
    pit1, pit2, _ = best_2
    assert 1 <= pit1 < pit2 < laps
    assert 1 <= best_1[0] < laps
    assert np.isfinite(best_time_2)


def test_run_simulation_rejects_very_short_race():
    with pytest.raises(ValueError, match="at least 10 laps"):
        run_simulation((sim_config(total_laps=9), 0))


def test_run_simulation_certain_failure_is_dnf():
    best_time_1, best_time_2, _, _, dnf_count = run_simulation((sim_config(mechanical_failure_prob=1.0), 0))
    assert dnf_count == 1
    assert best_time_1 == np.inf and best_time_2 == np.inf


def test_run_simulation_dnf_rate_matches_per_lap_chance():
    # 2% per lap over 30 laps: 1 - 0.98**30 = 0.455 of races. seeds are fixed,
    # so this always gives the same count; the range allows for 300 samples
    config = sim_config(mechanical_failure_prob=0.02, n_strategies_sampled=10)
    dnfs = sum(run_simulation((config, seed))[4] for seed in range(300))
    assert 0.38 < dnfs / 300 < 0.53


def calm_race(monkeypatch, **overrides):
    # no SC/VSC, traffic or noise, so every strategy has one fixed time
    monkeypatch.setattr('monte_carlo.make_noise', lambda total_laps, rng: {
        'pit': np.zeros(total_laps), 'deg': np.zeros(total_laps), 'traffic': np.ones(total_laps)})
    return sim_config(sc_chance=0.0, vsc_chance=0.0, **overrides)


def all_times(config, two_stop):
    # brute force every strategy in the search range with simulate_race
    names = list(config['compound_data'])
    laps = config['total_laps']
    green = np.zeros(laps, dtype=int)
    noise = {'pit': np.zeros(laps), 'deg': np.zeros(laps), 'traffic': np.ones(laps)}
    times = {}
    if two_stop:
        for p1 in range(1, laps - 5):
            for p2 in range(p1 + 1, laps - 2):
                for comps in [(a, b, c) for a in names for b in names for c in names if len({a, b, c}) >= 2]:
                    times[(p1, p2, comps)] = simulate_race(p1, comps, green, config, pit_lap2=p2, noise=noise)['total_time']
    else:
        for p in range(1, laps - 5):
            for comps in [(a, b) for a in names for b in names if a != b]:
                times[(p, comps)] = simulate_race(p, comps, green, config, noise=noise)['total_time']
    return times


@pytest.mark.parametrize('method', ['sampled', 'exhaustive'])
def test_run_simulation_finds_fastest_one_stop(monkeypatch, method):
    config = calm_race(monkeypatch, search_method=method)
    times = all_times(config, two_stop=False)
    best_time_1, _, best_1, _, _ = run_simulation((config, 0))
    assert best_time_1 == pytest.approx(min(times.values()))
    assert times[best_1] == pytest.approx(best_time_1)


def test_exhaustive_finds_fastest_two_stop(monkeypatch):
    config = calm_race(monkeypatch, search_method='exhaustive')
    times = all_times(config, two_stop=True)
    _, best_time_2, _, best_2, _ = run_simulation((config, 0))
    assert best_time_2 == pytest.approx(min(times.values()))
    assert times[best_2] == pytest.approx(best_time_2)


@pytest.mark.parametrize('extra', [{}, {'fuel_wear': 0.3, 'fuel_effect': 0.03}])
def test_calm_optimum_is_fastest_on_calm_race(extra):
    config = sim_config(**extra)
    one_times = all_times(config, two_stop=False)
    two_times = all_times(config, two_stop=True)
    best_1, best_2 = calm_optimum(config)
    assert one_times[best_1] == pytest.approx(min(one_times.values()))
    assert two_times[best_2] == pytest.approx(min(two_times.values()))


def test_sampled_two_stop_close_to_fastest(monkeypatch):
    # sampling plus refinement isn't guaranteed to hit the exact best, but on a
    # race with no noise it should be within a second of it
    config = calm_race(monkeypatch, search_method='sampled')
    fastest = min(all_times(config, two_stop=True).values())
    _, best_time_2, _, _, _ = run_simulation((config, 0))
    assert fastest <= best_time_2 + 1e-9
    assert best_time_2 - fastest < 1.0


# summarize

ONE_SH = (20, ('soft', 'hard'))
ONE_MH = (21, ('medium', 'hard'))
TWO_SMM = (15, 35, ('soft', 'medium', 'medium'))
TWO_SMM_B = (16, 36, ('soft', 'medium', 'medium'))
TWO_MHH = (14, 34, ('medium', 'hard', 'hard'))


def sample_results():
    return [
        (5000.0, 5010.0, ONE_SH, TWO_SMM, 0),                 # 1-stop wins
        (5000.0, 4990.0, ONE_MH, TWO_SMM_B, 0),               # 2-stop wins
        (np.inf, np.inf, (1, ('soft', 'medium')), (1, 2, ('soft', 'medium', 'hard')), 1),  # DNF
        (4995.0, 4995.0, ONE_SH, TWO_MHH, 0),                 # tie goes to 1-stop
    ]


def test_summarize_counts_and_skips_dnfs():
    s = summarize(sample_results(), 60)
    assert (s['n_sims'], s['total_dnfs'], s['n_finished']) == (4, 1, 3)
    assert s['winner'] == ['1-stop', '2-stop', '1-stop']
    assert (s['wins_1'], s['wins_2']) == (2, 1)
    assert s['most_common'] == '1-stop'
    assert s['pit_1'] == [20, 21, 20]
    assert s['pit_2a'] == [15, 16, 14]


def test_summarize_average_times_ignore_dnfs():
    s = summarize(sample_results(), 60)
    assert s['avg_time_1'] == pytest.approx(14995.0 / 3)
    assert s['avg_time_2'] == pytest.approx(14995.0 / 3)


def test_summarize_top_tables_only_count_wins():
    s = summarize(sample_results(), 60)
    assert s['top_1'] == [(ONE_SH, 2)]   # ONE_MH was best 1-stop once but lost to the 2-stop
    assert s['top_2'] == [(TWO_SMM_B, 1)]


def test_summarize_splits_pit_laps_by_compound():
    s = summarize(sample_results(), 60)
    assert s['by_compound']['1-stop'] == {'soft': [20, 20], 'medium': [21], 'hard': []}
    assert s['by_compound']['2-stop first'] == {'soft': [15, 16], 'medium': [14], 'hard': []}
    # second stop is split by the second stint's compound
    assert s['by_compound']['2-stop second'] == {'soft': [], 'medium': [35, 36], 'hard': [34]}


def test_summarize_representative_is_top_winner():
    s = summarize(sample_results(), 60)
    assert s['rep_1'] == ([20], ['soft', 'hard'])
    assert s['rep_2'] == ([16, 36], ['soft', 'medium', 'medium'])


def test_summarize_representative_tie_goes_to_faster_average():
    results = [
        (5005.0, 5100.0, ONE_SH, TWO_SMM, 0),
        (5001.0, 5100.0, ONE_MH, TWO_SMM, 0),
    ]
    s = summarize(results, 60)
    assert s['rep_1'] == ([21], ['medium', 'hard'])


def test_summarize_falls_back_to_even_split_when_type_never_won():
    results = [
        (5000.0, 5100.0, ONE_SH, TWO_MHH, 0),
        (5000.0, 5100.0, ONE_SH, TWO_MHH, 0),
        (5000.0, 5100.0, ONE_SH, TWO_SMM, 0),
    ]
    s = summarize(results, 60)
    assert s['wins_2'] == 0
    assert s['top_2'] == []
    assert s['rep_2'] == ([20, 40], ['medium', 'hard', 'hard'])   # most common best 2-stop compounds


def test_summarize_raises_when_every_run_dnfs():
    results = [(np.inf, np.inf, ONE_SH, TWO_SMM, 1)] * 3
    with pytest.raises(ValueError, match="DNF"):
        summarize(results, 60)


def test_compound_label():
    assert compound_label(('soft', 'medium', 'hard')) == 'S → M → H'


# simulation count and margins

def test_dnf_rate_from_per_lap_chance():
    # 1 - 0.998**60 = 1 - 0.88682
    assert dnf_rate(0.002, 60) == pytest.approx(0.11318, abs=1e-5)
    assert dnf_rate(0.0, 60) == 0.0


@pytest.mark.parametrize('margin, dnf, expected', [
    (0.03, 0.0, 1068),   # (1.96 / 0.03)**2 / 4 = 1067.1
    (0.05, 0.0, 385),    # (1.96 / 0.05)**2 / 4 = 384.2
    (0.03, 0.5, 2135),   # half the runs lost: 1067.1 / 0.5 = 2134.2
])
def test_sims_needed(margin, dnf, expected):
    assert sims_needed(margin=margin, dnf=dnf) == expected


def test_win_share_margin_even_split():
    # Wilson half-width at p = 0.5, n = 1000: 1.96 * sqrt(0.00025 + 9.604e-7) / 1.0038416
    assert win_share_margin(500, 1000) == pytest.approx(0.030931, abs=1e-5)


def test_win_share_margin_stays_above_zero_when_one_side_wins_everything():
    # plain p(1-p) would give 0; Wilson gives 1.96 * 0.00098 / 1.0038416
    assert win_share_margin(1000, 1000) == pytest.approx(0.0019134, abs=1e-6)


def test_win_share_margin_no_finished_races():
    assert win_share_margin(0, 0) == 1.0
