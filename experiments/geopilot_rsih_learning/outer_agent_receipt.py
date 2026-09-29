"""Complete, reproducible receipts for one dispatched four-arm Agent episode.

This module never calls a model, never runs the numerical chain and never reads
reference geometry. It only records, validates, projects and hashes what the
host dispatcher in ``outer_agent_dispatch`` observed. Every quantity that was
not observed stays ``None``; it is never zero-filled and never inferred.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RECEIPT_SCHEMA = 'geopilot-outer-agent-receipt/1'

# The six measures the frozen development scorer publishes, and nothing else.
METRICS = (
    'accuracy_l1_m', 'accuracy_rmse_m', 'accuracy_best90_l1_m',
    'accuracy_best90_rmse_m', 'precision_0_20', 'completeness_0_20',
)
# The complete set of keys that may cross the host/agent boundary. A score
# receipt is projected onto exactly this set before the agent ever sees it.
FEEDBACK_WHITELIST = frozenset({'status', 'surface_metrics', 'reason',
                                'rejected_config_digests'})

RECEIPT_KEYS = frozenset({
    'schema', 'arm', 'arm_label', 'arm_sha256', 'task', 'task_sha256',
    'capability', 'capability_sha256', 'capability_invariant', 'instruction',
    'instruction_sha256', 'model', 'turns', 'turn_count', 'tool_calls',
    'executed_actions', 'rejected_actions', 'feedback_history', 'status',
    'stop_reason', 'cost', 'timing', 'error',
})
TURN_KEYS = frozenset({'step_index', 'request', 'response', 'model_identity',
                       'usage', 'timing', 'proposal', 'outcome', 'reason'})
TOOL_CALL_KEYS = frozenset({'index', 'step_index', 'name', 'arguments',
                            'arguments_sha256', 'config', 'config_id',
                            'registered', 'argv', 'numerical', 'score',
                            'feedback', 'exit_status', 'elapsed_s'})
EXIT_STATUS = ('ok', 'failed', 'rejected', 'not_executed')
# Fixed, non-diagnostic error vocabulary. Provider-controlled message text is
# never persisted, so a reader cannot be steered by a remote string.
ERRORS = frozenset({
    'response_fault', 'body_incomplete', 'model_identity_missing',
    'model_identity_mismatch', 'finish_reason_not_stop', 'invalid_schema',
    'unregistered_config', 'unknown_tool', 'invalid_arguments', 'step_budget',
    'call_budget', 'run_budget', 'rejection_budget', 'time_budget', 'numerical_failed',
    'scoring_failed', 'delivery_invalid', 'send_gate', 'rule_table_exhausted',
})
USAGE_FIELD_MAP = {'input_tokens': 'prompt_tokens', 'output_tokens': 'completion_tokens'}
"""Provider field aliases M3 uses for the canonical usage names."""

PRIVATE = ('api_key', 'bearer', 'evaluator-only', 'protocol_v1', 'benchmark.py',
           'reference_manifest', '/labels', 'secret', 'password')


def sha256_bytes(data: bytes) -> str:
    """Return the hex digest of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    """Return the canonical JSON form used for every identity digest."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    """Return the sha256 of the canonical JSON form of a value."""
    return sha256_bytes(canonical(value).encode())


def sha256_file(path: Path) -> str:
    """Return the sha256 of an existing file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


def write_receipt(path: Path, value: Any) -> str:
    """Write one receipt with exclusive creation; never overwrite a record."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return sha256_file(path)


# Longest match first, alphabetical on ties. ERRORS is a frozenset and set
# iteration order varies with PYTHONHASHSEED, so scanning it directly would let
# hash randomisation decide which vocabulary term a receipt records.
ERRORS_BY_SPECIFICITY = tuple(sorted(ERRORS, key=lambda term: (-len(term), term)))


def normalise_error(exc: BaseException) -> str:
    """Collapse an exception to the fixed vocabulary; remote text never persists.

    A wrapped message can carry both a specific and a generic term; the longest
    match wins so the receipt records the specific cause. The scan order is
    fixed, so the same exception always yields the same label.
    """
    text = str(exc)
    for candidate in ERRORS_BY_SPECIFICITY:
        if candidate in text:
            return candidate
    return 'invalid_schema'


def require_sendable(payload: Any, *, forbidden: Sequence[str] = ()) -> None:
    """Refuse to put evaluator references, credentials or sibling arms on the wire.

    This is a construction-time gate, not a promise: it runs on the exact
    object handed to the transport, so a leak cannot be introduced later by a
    different call site.
    """
    lowered = tuple(p.lower() for p in forbidden)

    def blocked(text: str) -> bool:
        low = text.lower()
        return any(bad in low for bad in PRIVATE) or any(p in low for p in lowered)

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if not isinstance(key, str) or blocked(key) or 'token' in key.lower():
                    raise ValueError('send_gate')
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, str) and blocked(node):
            raise ValueError('send_gate')

    walk(payload)
    return None


def _pick(records: Sequence[Mapping[str, Any]], name: str,
          candidates: Sequence[str]) -> tuple[int | None, str | None]:
    """Sum one canonical usage field over per-call records; absent stays None."""
    values: list[int] = []
    found = None
    for record in records:
        for candidate in candidates:
            if candidate in record:
                value = record[candidate]
                if type(value) is not int or value < 0:
                    return None, None
                values.append(value)
                found = candidate if candidate != name else None
                break
        else:
            return None, None
    return sum(values), found


def normalise_usage(raw: Any) -> dict:
    """Map provider-reported usage onto one canonical shape; absences stay null.

    MiniMax-M3 reports ``prompt_tokens``/``completion_tokens``; M3.1 reports
    ``input_tokens``/``output_tokens`` and, on tool continuations, an array of
    per-call usage objects. Both shapes are accepted and the raw value is kept.
    """
    records: list[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        records = [raw]
    elif isinstance(raw, (list, tuple)) and raw and all(isinstance(r, Mapping) for r in raw):
        records = list(raw)
    if not records:
        return {'input_tokens': None, 'output_tokens': None, 'total_tokens': None,
                'reasoning_tokens': None, 'cache_read_tokens': None,
                'source': 'unobserved', 'field_map': {}, 'usage_raw': raw}
    totals: dict[str, int | None] = {}
    field_map: dict[str, str] = {}
    for name, candidates in (('input_tokens', ('input_tokens', 'prompt_tokens')),
                             ('output_tokens', ('output_tokens', 'completion_tokens')),
                             ('total_tokens', ('total_tokens',))):
        totals[name], mapped = _pick(records, name, candidates)
        if mapped:
            field_map[name] = mapped
    return {'input_tokens': totals['input_tokens'], 'output_tokens': totals['output_tokens'],
            'total_tokens': totals['total_tokens'],
            'reasoning_tokens': _detail_total(records, 'output_token_details', 'reasoning'),
            'cache_read_tokens': _detail_total(records, 'input_token_details', 'cache_read'),
            'source': 'provider_self_reported', 'field_map': field_map, 'usage_raw': raw}


def _detail_total(records: Sequence[Mapping[str, Any]], source: str, key: str) -> int | None:
    """Sum an optional per-call detail field; never observed means never None-filled."""
    total = 0
    seen = False
    for record in records:
        details = record.get(source)
        if isinstance(details, Mapping) and key in details:
            value = details[key]
            if type(value) is not int or value < 0:
                return None
            total += value
            seen = True
    return total if seen else None


def dropped_keys(score: Mapping[str, Any],
                 whitelist: Iterable[str] = FEEDBACK_WHITELIST) -> list:
    """Return the names a score receipt loses to the whitelist, for the audit trail."""
    allowed = frozenset(whitelist)
    names = [key for key in sorted(score) if key not in allowed]
    values = score.get('surface_metrics')
    if isinstance(values, Mapping):
        names += ['surface_metrics.' + key for key in sorted(values) if key not in METRICS]
    return sorted(set(names))


def project_feedback(score: Mapping[str, Any], *, whitelist: Iterable[str] = FEEDBACK_WHITELIST,
                     projector: Any = None) -> dict:
    """Project a score receipt onto the whitelist; the receipt itself never leaves.

    A deliberately leaky receipt (reference manifests, reference-derived counts,
    extra metrics inside ``surface_metrics``) loses every non-whitelisted key
    here. Nothing about the drop is sent back; ``dropped_keys`` records it on the
    host side only.
    """
    allowed = frozenset(whitelist)
    if not isinstance(score, Mapping):
        raise ValueError('delivery_invalid')
    projected: dict[str, Any] = {}
    if 'surface_metrics' in score:
        values = score['surface_metrics']
        if not isinstance(values, Mapping):
            raise ValueError('delivery_invalid')
        metrics = {}
        for name in METRICS:
            value = values.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (name in METRICS[-2:] and value > 1)):
                raise ValueError('delivery_invalid')
            metrics[name] = value
        projected['surface_metrics'] = metrics
    if score.get('status') is not None:
        projected['status'] = score['status']
    if 'reason' in allowed and isinstance(score.get('reason'), str):
        projected['reason'] = score['reason']
    if 'rejected_config_digests' in allowed and isinstance(score.get('rejected_config_digests'), list):
        projected['rejected_config_digests'] = list(score['rejected_config_digests'])
    if set(projected) - allowed:
        raise ValueError('send_gate')
    if projector is not None and 'surface_metrics' in projected:
        # Cross-check against the authoritative projector when the host has one.
        if projector(score) != {'status': projected.get('status'), 'surface_metrics': metrics}:
            raise ValueError('delivery_invalid')
    return projected


def new_cost() -> dict:
    """Return an empty cost record; every observed quantity starts unobserved."""
    return {'model_calls': 0, 'input_tokens': None, 'output_tokens': None,
            'total_tokens': None, 'reasoning_tokens': None, 'cache_read_tokens': None,
            'usage_field_map': {}, 'usage_source': 'unobserved',
            'calls_without_usage': 0, 'numerical_runs': 0, 'score_calls': 0,
            'numerical_wall_clock_s': 0.0, 'score_wall_clock_s': 0.0,
            'wall_clock_s': 0.0, 'peak_rss_bytes': None, 'peak_rss_source': None,
            'gpu_seconds': None}


def _add_optional(target: dict, source: Mapping[str, Any], key: str) -> None:
    value = source.get(key)
    if type(value) is int and value >= 0:
        target[key] = target[key] + value if target.get(key) is not None else value


def add_usage(cost: dict, usage: Mapping[str, Any]) -> dict:
    """Accumulate one call's normalised usage; a missing field stays null."""
    cost['model_calls'] += 1
    for key in ('input_tokens', 'output_tokens', 'total_tokens',
                'reasoning_tokens', 'cache_read_tokens'):
        _add_optional(cost, usage, key)
    if usage.get('total_tokens') is None and usage.get('input_tokens') is not None \
            and usage.get('output_tokens') is not None:
        cost['total_tokens'] = cost['input_tokens'] + cost['output_tokens']
        cost['usage_field_map'] = {**cost['usage_field_map'],
                                   'total_tokens': 'derived_input_plus_output'}
    cost['usage_field_map'].update(usage.get('field_map') or {})
    if usage.get('source') == 'unobserved':
        cost['calls_without_usage'] += 1
    else:
        cost['usage_source'] = 'provider_self_reported'
    return cost


def build_receipt(*, arm: str, arm_label: str, arm_sha256: str, task: Mapping[str, Any],
                  capability: Mapping[str, Any], capability_invariant: str,
                  instruction: Mapping[str, Any], model: Mapping[str, Any],
                  turns: Sequence[Mapping[str, Any]], tool_calls: Sequence[Mapping[str, Any]],
                  executed_actions: Sequence[Any], rejected_actions: Sequence[Any],
                  feedback_history: Sequence[Any], status: str, stop_reason: str,
                  cost: Mapping[str, Any], timing: Mapping[str, Any],
                  error: str | None) -> dict:
    """Assemble a receipt with an exact key set; unknown keys cannot be added."""
    receipt = {
        'schema': RECEIPT_SCHEMA,
        'arm': arm,
        'arm_label': arm_label,
        'arm_sha256': arm_sha256,
        'task': dict(task),
        'task_sha256': digest(task),
        'capability': dict(capability),
        'capability_sha256': digest(capability),
        'capability_invariant': capability_invariant,
        'instruction': dict(instruction),
        'instruction_sha256': digest(instruction),
        'model': dict(model),
        'turns': [dict(t) for t in turns],
        'turn_count': len(turns),
        'tool_calls': [dict(c) for c in tool_calls],
        'executed_actions': list(executed_actions),
        'rejected_actions': list(rejected_actions),
        'feedback_history': list(feedback_history),
        'status': status,
        'stop_reason': stop_reason,
        'cost': dict(cost),
        'timing': dict(timing),
        'error': error,
    }
    require_receipt(receipt)
    return receipt


def require_receipt(receipt: Mapping[str, Any]) -> None:
    """Exact key-set and value discipline; the schema is never widened silently."""
    if not isinstance(receipt, Mapping) or set(receipt) != RECEIPT_KEYS:
        raise ValueError('invalid_schema')
    if receipt['schema'] != RECEIPT_SCHEMA or receipt['error'] not in ERRORS | {None}:
        raise ValueError('invalid_schema')
    if receipt['status'] not in ('completed', 'failed', 'invalid'):
        raise ValueError('invalid_schema')
    for turn in receipt['turns']:
        if set(turn) != TURN_KEYS or turn['outcome'] not in (
                'executed', 'rejected', 'stopped', 'terminated', 'rule_table_step'):
            raise ValueError('invalid_schema')
    for call in receipt['tool_calls']:
        if set(call) != TOOL_CALL_KEYS or call['exit_status'] not in EXIT_STATUS:
            raise ValueError('invalid_schema')
    if receipt['arm_sha256'] != digest(receipt['instruction']):
        raise ValueError('invalid_schema')
    return None
