"""Pure evidence bookkeeping, not a scientific truth or advantage evaluator.

Host integration: call ``transition`` after each actual tool return (rows omitted),
then expose the resulting needs_update rows to the next model turn. Explicit model
upserts clear that flag; G0/GE must resolve it before action/stop, L may use the same
operation optionally. The host, not this module, enforces arm policy.

Claim row keys (all required): claim_id, observation_refs, alternatives,
proposed_action, predicted_observation, observed_result_refs, status.
Refs are exact ``step-N`` (un-padded) IDs or successful diagnostic source IDs.
status: unresolved/supported/contradicted/not_tested. Predictions are strings,
observations are references only. A submitted row acknowledges currently returned
evidence; do not resubmit old rows automatically. Upserts never delete other rows.

Experience row keys: entry_id, applicability_refs, status (eligible/inapplicable/
unresolved). Supply the frozen entry list for GE, empty for G0/L. The host preserves
its original provenance. Conditions, counterevidence and unproven fields are copied
from the frozen entries, never supplied or overwritten by the model. No new memory
is promoted. Pass current source hashes (including None for unavailable sources) and
an execution-source digest as code_version on every transition to detect drift.

Only actual tool_calls/history feedback is indexed. Failures are observations, not
successful source reads. Do not pass model-invented history. Returned observations
contain digests/IDs only, so the host retains the original fact receipts separately.
"""
from __future__ import annotations

import copy
import hashlib
import json

CLAIM_KEYS = frozenset(('claim_id', 'observation_refs', 'alternatives', 'proposed_action',
                        'predicted_observation', 'observed_result_refs', 'status'))
EXPERIENCE_KEYS = frozenset(('entry_id', 'applicability_refs', 'status'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _refs(value, available):
    if not isinstance(value, list) or any(not isinstance(r, str) or r not in available for r in value):
        raise ValueError('unknown_or_stale_observation_reference')
    if len(value) != len(set(value)):
        raise ValueError('duplicate_reference')
    return copy.deepcopy(value)


def _catalog(tool_calls, history):
    catalog = {}
    for event in [*tool_calls, *history]:
        feedback = event.get('feedback')
        if feedback is None or event.get('name') == 'stop':
            continue
        step = event.get('step_index')
        if type(step) is not int or step < 0 or not isinstance(feedback, dict):
            raise ValueError('invalid_actual_observation')
        sources = {}
        if feedback.get('kind') == 'diagnostic' and feedback.get('status') == 'ok':
            # The dispatcher wraps a domain diagnostic receipt inside facts.
            inner = feedback.get('facts') or {}
            hashes = inner.get('source_sha256', feedback.get('sources', {}))
            for source, sha in hashes.items():
                if not isinstance(source, str) or source.startswith('step-'):
                    raise ValueError('invalid_source_id')
                if sha is not None:
                    if not isinstance(sha, str) or not sha:
                        raise ValueError('invalid_source_hash')
                    sources[source] = sha
        entry = {'feedback_sha256': _digest(feedback), 'sources': sources}
        key = 'step-' + str(step)
        if key in catalog and catalog[key] != entry:
            raise ValueError('conflicting_actual_observations')
        catalog[key] = entry
    return dict(sorted(catalog.items(), key=lambda item: int(item[0][5:])))


def transition(previous=None, *, tool_calls=(), history=(), rows=None,
               experience_rows=None, frozen_experience=(), source_versions=None,
               code_version=None):
    """Return a detached state; reject invented refs and preserve frozen history."""
    old = previous or {}
    catalog = _catalog(tool_calls, history)
    for key, value in old.get('observations', {}).items():
        if key not in catalog or catalog[key] != value:
            raise ValueError('actual_observation_history_rewritten')
    entries = {entry['id']: entry for entry in frozen_experience}
    if len(entries) != len(frozen_experience):
        raise ValueError('duplicate_frozen_experience_id')
    frozen_hash = _digest(list(frozen_experience))
    if old and old.get('frozen_experience_sha256') != frozen_hash:
        raise ValueError('frozen_experience_changed')
    current_sources = {}
    for value in catalog.values():
        current_sources.update(value['sources'])
    if source_versions is not None:
        current_sources.update(source_versions)
    stale_steps = {key for key, value in catalog.items()
                   if any(current_sources.get(s) != sha for s, sha in value['sources'].items())}
    available = set(catalog) - stale_steps
    for key, value in catalog.items():
        if key not in stale_steps:
            available.update(value['sources'])
    changed = (old.get('observations', {}) != catalog
               or old.get('source_versions', {}) != current_sources
               or old.get('code_version') != code_version)
    claims = copy.deepcopy(old.get('claims', {}))
    memories = copy.deepcopy(old.get('experience', {}))
    drifted = bool(old) and (old.get('code_version') != code_version or any(
        current_sources.get(key) != value for key, value in old.get('source_versions', {}).items()))
    for key, entry in entries.items():
        memories.setdefault(key, {'entry_id': key, 'applicability_refs': [], 'status': 'unresolved',
            'needs_update': True, **{field: copy.deepcopy(entry.get(field)) for field in (
                'conditions', 'counterevidence_and_boundaries', 'unproven')}})
    if changed:
        for row in [*claims.values(), *memories.values()]:
            row['needs_update'] = True
            if drifted:
                row['status'] = 'unresolved'
    seen = set()
    for row in rows or []:
        if not isinstance(row, dict) or set(row) != CLAIM_KEYS:
            raise ValueError('invalid_claim_row')
        key = row['claim_id']
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('invalid_or_duplicate_claim_id')
        seen.add(key)
        if row['status'] not in ('unresolved', 'supported', 'contradicted', 'not_tested'):
            raise ValueError('invalid_claim_status')
        for field in ('alternatives', 'proposed_action', 'predicted_observation'):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError('invalid_claim_text')
        observations = _refs(row['observation_refs'], available)
        results = _refs(row['observed_result_refs'], available)
        if row['status'] in ('supported', 'contradicted') and not results:
            raise ValueError('observed_result_required')
        if row['status'] == 'not_tested' and results:
            raise ValueError('not_tested_with_observed_result')
        existing = claims.get(key)
        if existing and results and row['predicted_observation'] != existing['predicted_observation']:
            raise ValueError('prediction_changed_during_result_backfill')
        claims[key] = {**copy.deepcopy(row), 'observation_refs': observations,
                       'observed_result_refs': results, 'needs_update': False}
    seen = set()
    for row in experience_rows or []:
        if not isinstance(row, dict) or set(row) != EXPERIENCE_KEYS:
            raise ValueError('invalid_experience_row')
        key = row['entry_id']
        if key not in entries or key in seen:
            raise ValueError('unknown_or_duplicate_frozen_entry')
        seen.add(key)
        if row['status'] not in ('eligible', 'inapplicable', 'unresolved'):
            raise ValueError('invalid_experience_status')
        refs = _refs(row['applicability_refs'], available)
        if row['status'] != 'unresolved' and not refs:
            raise ValueError('applicability_evidence_required')
        entry = entries[key]
        memories[key] = {**copy.deepcopy(row), 'applicability_refs': refs, 'needs_update': False,
                         **{field: copy.deepcopy(entry.get(field)) for field in (
                             'conditions', 'counterevidence_and_boundaries', 'unproven')}}
    return {'schema': 'geopilot-decision-state/1', 'observations': catalog,
            'source_versions': current_sources, 'code_version': code_version,
            'stale_observation_ids': sorted(stale_steps), 'claims': claims,
            'experience': memories, 'frozen_experience_sha256': frozen_hash}
