"""Check the public scalar/knowledge snapshot; no geometry or network access."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('accuracy_l1_m', 'accuracy_rmse_m', 'accuracy_best90_l1_m',
           'accuracy_best90_rmse_m', 'precision_0_20', 'completeness_0_20')


def main():
    source = json.loads((ROOT / 'evidence/measurements.json').read_text())
    records = {r['id']: r for r in source['records']}
    assert len(source['records']) == len(records) == 7
    assert set(records) == {'P0', 'P1', 'P2', 'A', 'B', 'S-U', 'F-U'}
    for record in records.values():
        assert record['status'] == 'valid'
        for field in ('source_score_sha256', 'source_manifest_sha256'):
            value = record[field]
            assert len(value) == 64 and all(c in '0123456789abcdef' for c in value)
        for name in METRICS:
            value = record['surface_metrics'][name]
            assert isinstance(value, (float, int)) and math.isfinite(value) and value >= 0
            if name in METRICS[4:]:
                assert value <= 1
    a, b = (records[k]['surface_metrics'] for k in ('A', 'B'))
    for name in METRICS[:4]:
        assert b[name] < a[name]
    for name in METRICS[4:]:
        assert b[name] > a[name]
    assert math.isclose(b[METRICS[0]] - a[METRICS[0]], -0.139982802281, abs_tol=1e-10)
    assert math.isclose(100 * (b[METRICS[4]] - a[METRICS[4]]), 1.4462, abs_tol=1e-8)
    assert math.isclose(100 * (b[METRICS[5]] - a[METRICS[5]]), 0.7048799171, abs_tol=1e-8)
    knowledge = json.loads((ROOT / 'knowledge/tool-knowledge.json').read_text())
    assert len({x['id'] for x in knowledge['entries']}) == len(knowledge['entries'])
    for item in knowledge['entries']:
        assert item['type'] in {'fact', 'hypothesis', 'open'}
        assert item['claim'] and item['source']
    for name in ('shared-intrinsics.md', 'program-outcomes.md'):
        assert (ROOT / 'knowledge' / name).is_file()
    diagnostic = json.loads((ROOT / 'evidence/alignment-sensitivity.json').read_text())
    assert diagnostic['formal_score'] is False and diagnostic['reference_used_for_fitting'] is True
    assert len(diagnostic['source_sha256']) == 64
    rows = {(r['arm'], r['alignment']): r for r in diagnostic['records']}
    families = ('identity', 'translation', 'rigid', 'similarity')
    assert len(rows) == len(diagnostic['records']) == 8
    assert set(rows) == {(arm, kind) for arm in ('A', 'B') for kind in families}
    for row in rows.values():
        assert row['fit_samples'] == 6000 and row['validation_samples'] == 12000
        assert math.isfinite(row['validation_mean_m']) and row['validation_mean_m'] >= 0
    for kind in families:
        difference = rows['B', kind]['validation_mean_m'] - rows['A', kind]['validation_mean_m']
        assert difference < 0 if kind == 'identity' else difference > 0
    print('PASS: 8 separately labeled alignment-diagnostic records and ranking directions')
    print('PASS: 7 metric records, shared-intrinsics deltas, knowledge types and experience records')
    print('Scope: published scalar consistency, not independent geometry recomputation.')


if __name__ == '__main__':
    main()
