"""The one place a numerical configuration knob may be extended.

`REGISTERED_CONFIGS` is closed frozen data: stable ids, explicit values, no wildcards,
no ranges and no "merge whatever the caller passes". `require_registered()` raises
unless the requested configuration is inside that set, and `resolve()` returns the full
effective parameters dict for it. `config=None` resolves to `BASELINE`, the parameter set
that every recorded result before this module was written was produced under.

Only a key whose wiring is PROVEN may appear in `REGISTERED_CONFIGS`. The verdicts below
are copied from the static trace
`out/geopilot-research-20260929/four-arm-executor/trace/1-parameter-wiring.md`
(static read-and-trace; no reconstruction, dense, mesh or evaluator job was run there or
here). A key marked DEAD or UNVERIFIED is documented as unusable and is rejected by
`_self_check()` at import time, so a knob that merely *looks* wired cannot silently
produce an experiment in which every arm runs the same thing.
"""
import copy

SCHEMA = 'geopilot-outer-config/1'
BASELINE_ID = 'baseline-v1'

#: Today's frozen effective parameters, i.e. ``outer_numerical.PARAMETERS`` and
#: ``outer_score.PARAMETERS``. It is repeated here only so this module stays importable
#: without pycolmap; both of those modules assert equality against it at import, so the
#: three copies cannot drift silently.
BASELINE = {
    'undistort_max_image_size': 2400,
    'densify': {'resolution-level': 1},
    'mesh': {'decimate': 0.5},
    'undistort_call': {'output_type': 'COLMAP', 'copy_policy': 'copy', 'num_patch_match_src_images': 20},
    'transform_application': 'once at native OBJ to world PLY export; never transform source model',
    'export_atol_m': 1e-9,
    'export_rtol': 0.0,
}

#: Per-key wiring verdict. `consumed_at` names the exact place the value is read on the
#: outer path; `reason` says what happens when the key is changed.
KEY_VERDICTS = {
    'mesh.decimate': {
        'verdict': 'PROVEN', 'live': True, 'action': 'mesh',
        'consumed_at': 'outer_numerical.py mesh_worker -> request.json node.parameters '
                       '-> code/geopilot_rsih/tools.py argv',
        'reason': 'Shipped ReconstructMesh default is 1 ("1 - disabled"); 0.5 already '
                  'changes the delivered mesh, so this is the only live key today.'},
    'densify.resolution-level': {
        'verdict': 'PROVEN', 'live': False, 'action': 'densify',
        'consumed_at': 'outer_numerical.py mesh_worker -> request.json node.parameters '
                       '-> code/geopilot_rsih/tools.py argv',
        'reason': 'Reaches argv and is a real option of the bound DensifyPointCloud, but '
                  'the shipped default is 1, so the registered baseline value is inert. '
                  'Values other than 1 are live.'},
    'undistort_max_image_size': {
        'verdict': 'DEAD', 'live': False, 'action': None,
        'consumed_at': 'none; the real call is driver.undistort_options(), a hardcoded '
                       'pc.UndistortCameraOptions(max_image_size=2400)',
        'reason': 'Changes a recorded JSON field and nothing else. Not registrable.'},
    'undistort_call': {
        'verdict': 'DEAD', 'live': False, 'action': None,
        'consumed_at': 'none; mesh_worker passes output_type/copy_policy/'
                       'num_patch_match_src_images as literals',
        'reason': 'Splatted into a written JSON only. Not registrable.'},
    'transform_application': {
        'verdict': 'DEAD', 'live': False, 'action': None,
        'consumed_at': 'none; zero read sites',
        'reason': 'Documentation string. Not registrable.'},
    'export_atol_m': {
        'verdict': 'CONSUMED_NOT_GEOMETRIC', 'live': False, 'action': None,
        'consumed_at': 'driver.verify_export, called from outer_numerical.py mesh_worker',
        'reason': 'The tolerance of the export-equality gate. Changing it can only relax '
                  'a safety check, never the mesh. Excluded by rule.'},
    'export_rtol': {
        'verdict': 'DEAD', 'live': False, 'action': None,
        'consumed_at': 'none; zero read sites',
        'reason': 'Defined and never read. Not registrable.'},
}

#: Keys that may be overridden at all. Everything else is dead, documentary or a gate.
REGISTRABLE = ('mesh.decimate', 'densify.resolution-level')

#: Densify key whose non-default value is live. Trace 1 calls level 0 roughly four times
#: the depth-map work against the 4 h / 48 GiB per-worker guard; level 2 is the cheap
#: direction. Verification of the cost is a later phase, not this one.
LIVE_RESOLUTION_LEVELS = (0, 2)


def _with(base, path, value):
    """Construction-time only: build one frozen registered entry from BASELINE."""
    head, _, tail = path.partition('.')
    result = copy.deepcopy(base)
    if tail:
        result[head] = _with(result[head], tail, value)
    else:
        result[head] = value
    return result


REGISTERED_CONFIGS = (
    {'id': BASELINE_ID, 'parameters': copy.deepcopy(BASELINE), 'knobs': ()},
    {'id': 'mesh-decimate-025-v1',
     'parameters': _with(BASELINE, 'mesh.decimate', 0.25),
     'knobs': ('mesh.decimate',)},
    {'id': 'densify-resolution-2-v1',
     'parameters': _with(BASELINE, 'densify.resolution-level', 2),
     'knobs': ('densify.resolution-level',)},
)


def _entry(config_id: str) -> dict:
    for entry in REGISTERED_CONFIGS:
        if entry['id'] == config_id:
            return entry
    raise ValueError('Unregistered configuration id: ' + config_id)


def changed_keys(parameters: dict, base: dict = None) -> tuple:
    """Dotted keys where `parameters` differs from `base` (BASELINE by default)."""
    base = BASELINE if base is None else base
    changed = []
    for key in sorted(set(parameters) | set(base)):
        if key not in parameters or key not in base:
            changed.append(key)
        elif isinstance(parameters[key], dict) and isinstance(base[key], dict):
            changed += [key + '.' + name for name in changed_keys(parameters[key], base[key])]
        elif type(parameters[key]) is not type(base[key]) or parameters[key] != base[key]:
            changed.append(key)
    return tuple(changed)


def _subsumes(registered: dict, override) -> bool:
    """True when `override` is a non-empty deep subset of one registered dict."""
    if not isinstance(override, dict) or not override:
        return False
    for key, value in override.items():
        if key not in registered:
            return False
        if isinstance(value, dict):
            if not isinstance(registered[key], dict) or not _subsumes(registered[key], value):
                return False
        elif isinstance(value, bool) or isinstance(registered[key], bool) \
                or type(value) is not type(registered[key]) or value != registered[key]:
            return False
    return True


def require_registered(config=None) -> str:
    """Return the canonical id of `config`; raise unless it is inside the registered set.

    Accepts None, a registered id, {'id': <registered id>}, a full registered parameter
    dict, or a non-empty deep subset of exactly one registered parameter dict.
    """
    if config is None:
        return BASELINE_ID
    if isinstance(config, str):
        if not config:
            raise ValueError('Empty configuration id')
        return _entry(config)['id']
    if isinstance(config, dict) and set(config) == {'id'}:
        if not isinstance(config['id'], str):
            raise ValueError('Configuration id must be a string')
        return _entry(config['id'])['id']
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a registered id or a dict')
    matches = [entry for entry in REGISTERED_CONFIGS if _subsumes(entry['parameters'], config)]
    if len(matches) == 1:
        return matches[0]['id']
    if not matches:
        raise ValueError('Configuration is not inside the registered set')
    raise ValueError('Configuration is ambiguous between registered ids: '
                     + ', '.join(sorted(entry['id'] for entry in matches)))


def resolve(config_or_id=None) -> dict:
    """Full effective parameters for a registered configuration, `BASELINE` when None."""
    return copy.deepcopy(_entry(require_registered(config_or_id))['parameters'])


def argv_flags(parameters: dict) -> list:
    """Exactly the flags the frozen worker appends, per code/geopilot_rsih/tools.py.

    Kept here so a knob can be proven to reach argv offline; the real worker calls the
    frozen assembler, this is the same rule and nothing else.
    """
    flags = []
    for key, value in sorted(parameters.items()):
        flags += ['--' + key, str(value)]
    return flags


def audit_wiring(config=None) -> str:
    """Raise unless every knob of a registered configuration still reaches argv.

    This is the loud-failure path for a knob that is later found dead: a registered
    entry whose knob is not PROVEN, or whose flag no longer appears in the argv the
    worker will assemble, stops the run instead of producing a second arm identical to
    the first under a different label.
    """
    config_id = require_registered(config)
    entry = _entry(config_id)
    knobs = changed_keys(entry['parameters'])
    if knobs != entry['knobs']:
        raise ValueError('Registered configuration does not match its declared knobs: ' + config_id)
    for knob in knobs:
        verdict = KEY_VERDICTS.get(knob)
        if verdict is None or verdict['verdict'] != 'PROVEN':
            raise ValueError('Knob is not PROVEN to reach the numerical chain: ' + knob)
        action = verdict['action']
        leaf = knob.split('.', 1)[1]
        flags = () if action is None else argv_flags(entry['parameters'][action])
        if action is None or '--' + leaf not in flags:
            raise ValueError('Knob does not reach the assembled worker argv: ' + knob)
        if knob == 'densify.resolution-level' and entry['parameters'][action][leaf] \
                not in LIVE_RESOLUTION_LEVELS:
            raise ValueError('Registered resolution level is the inert binary default: ' + knob)
    return config_id


def _self_check() -> None:
    """Refuse to import a registry that is not closed, PROVEN-only and non-degenerate."""
    ids = [entry['id'] for entry in REGISTERED_CONFIGS]
    if len(ids) != len(set(ids)) or BASELINE_ID not in ids:
        raise ValueError('Registered configuration ids must be unique and include the baseline')
    if [entry['parameters'] for entry in REGISTERED_CONFIGS if entry['id'] == BASELINE_ID][0] != BASELINE:
        raise ValueError('The registered baseline is not BASELINE')
    for entry in REGISTERED_CONFIGS:
        audit_wiring(entry['id'])
        knobs = changed_keys(entry['parameters'])
        if knobs and not set(knobs) <= set(REGISTRABLE):
            raise ValueError('Unregistrable knob in ' + entry['id'] + ': ' + ', '.join(knobs))
        if entry['id'] != BASELINE_ID and not knobs:
            raise ValueError('A registered configuration must differ from the baseline: ' + entry['id'])
    if len(REGISTERED_CONFIGS) < 2:
        raise ValueError('The registered set must offer at least one action, not a single point')


_self_check()
