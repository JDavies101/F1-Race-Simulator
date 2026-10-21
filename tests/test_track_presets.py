# Tests for loading fitted values over the hand-set presets.

import json

from track_presets import apply_fitted


def presets():
    return {
        'Custom': None,
        'Track A': {'base_lap_time': 90, 'soft_offset': -0.6, 'hard_deg': 0.02},
        'Track B': {'base_lap_time': 80, 'soft_offset': -0.6, 'hard_deg': 0.02},
    }


def write(tmp_path, data):
    path = tmp_path / 'fitted.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_fitted_values_override_hand_set_ones(tmp_path):
    p = presets()
    updated = apply_fitted(p, write(tmp_path, {'Track A': {'soft_offset': -0.16, 'hard_deg': 0.023}}))
    assert updated == ['Track A']
    assert p['Track A'] == {'base_lap_time': 90.0, 'soft_offset': -0.16, 'hard_deg': 0.023}
    assert p['Track B'] == {'base_lap_time': 80.0, 'soft_offset': -0.6, 'hard_deg': 0.02}   # untouched


def test_unknown_keys_are_ignored_but_fuel_is_allowed(tmp_path):
    p = presets()
    apply_fitted(p, write(tmp_path, {'Track A': {'hard_dge': 0.5, 'fuel_effect': 0.035}}))
    assert 'hard_dge' not in p['Track A']
    assert p['Track A']['hard_deg'] == 0.02
    assert p['Track A']['fuel_effect'] == 0.035


def test_unknown_and_empty_presets_are_skipped(tmp_path):
    p = presets()
    updated = apply_fitted(p, write(tmp_path, {'Nowhere': {'hard_deg': 0.5}, 'Custom': {'hard_deg': 0.5}}))
    assert updated == []
    assert p['Custom'] is None
    assert 'Nowhere' not in p


def test_missing_file_keeps_presets(tmp_path):
    p = presets()
    assert apply_fitted(p, tmp_path / 'does_not_exist.json') == []
    assert p['Track A']['soft_offset'] == -0.6


def test_base_lap_time_becomes_float(tmp_path):
    p = presets()
    apply_fitted(p, tmp_path / 'does_not_exist.json')
    assert isinstance(p['Track A']['base_lap_time'], float)
    assert p['Track A']['base_lap_time'] == 90.0
