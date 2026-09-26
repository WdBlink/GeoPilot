"""Exploratory C: shared-camera BA with only principal-point refinement enabled."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import shared_camera_ba_diagnostic as base

SOURCE_SHA = '07d85983a366f8b928e104a47884d9ddaea343554efd3b0c0b0210db9c5f71ac'
CONTROL = base.OUTPUT
OUTPUT = base.BASE / 'shared-principal-point-ba-run'
PLAN = base.BASE / 'shared-principal-point-ba-plan.md'
CONTROL_SHA = {
    'inputs.json': 'd6a292e04684544845919e880f18625a3726e4290f156645eddb3361dc53682c',
    'B/result.json': 'b5750c704304ef47df1d9151a9be7f5a397ef8360441f737eafdb862c42d6c24',
    'B/solver-settings.json': '7d180581a72b8eded93779d8e920493347e4a40260b82c2f5bfa104636a1a535',
    'B/supervision.json': '02835d6e49ea0ea9017460dca3e361fb02256429c2a810104bd2ef2e08f787ba',
}
base.verify({str(Path(base.__file__).resolve()): SOURCE_SHA,
             **{str(CONTROL / name): digest for name, digest in CONTROL_SHA.items()}})
B_SETTINGS = base.read_json(CONTROL / 'B/solver-settings.json')
ORIGINAL_OPTIONS, ORIGINAL_BRANCH = base.options, base.branch_model


def json_value(value):
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def options():
    """Require the frozen B defaults, then change this one boolean only."""
    value = ORIGINAL_OPTIONS()
    if json_value(value.todict()) != B_SETTINGS['options'] or B_SETTINGS['options']['refine_principal_point'] is not False:
        raise ValueError('Current solver defaults differ from the completed B control')
    value.refine_principal_point = True
    expected = {**B_SETTINGS['options'], 'refine_principal_point': True}
    if json_value(value.todict()) != expected:
        raise ValueError('C must change only refine_principal_point')
    return value


def branch_model(source, name, initial):
    if name != 'C':
        raise ValueError('Only the fixed C branch is permitted')
    return ORIGINAL_BRANCH(source, 'B', initial)


# These three process-local overrides are applied in both parent and child.
# The frozen implementation on disk and its existing campaign are untouched.
base.OUTPUT, base.options, base.branch_model = OUTPUT, options, branch_model


def inputs():
    """Include every old input binding and the complete B evidence being reused."""
    if Path(sys.prefix).resolve() != base.RUNTIME.resolve() or base.pc.__version__ != '3.12.6':
        raise RuntimeError('Use the fixed baseline-runtime Python with pycolmap 3.12.6')
    prior = base.read_json(CONTROL / 'inputs.json')
    result = base.read_json(CONTROL / 'B/result.json')
    supervision = base.read_json(CONTROL / 'B/supervision.json')
    if (result['status'] != 'completed' or result['branch'] != 'B' or not result['solution_usable']
            or supervision['returncode'] != 0 or supervision['reason'] is not None
            or prior['limits'] != base.LIMITS or prior['initial_K'] != base.INITIAL_K):
        raise ValueError('The completed B control or original initialization is invalid')
    bound = {**prior['bindings'], **result['copied_input_hashes'], **result['output_hashes'],
             **supervision['final_log_hashes'],
             **{str(CONTROL / name): digest for name, digest in CONTROL_SHA.items()}}
    bound[str(Path(base.__file__).resolve())] = SOURCE_SHA
    for path in (Path(__file__).resolve(), PLAN):
        bound[str(path)] = base.sha(path)
    base.verify(bound)
    return {**prior, 'bindings': bound, 'options': json_value(options().todict()),
            'exploratory_after_observed_results': True, 'formal_candidate': False,
            'control_B': str(CONTROL / 'B'), 'changed_option': {'refine_principal_point': [False, True]}}


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', '_worker'))
    args = parser.parse_args()
    manifest = inputs()
    if args.action == 'check':
        if manifest['options'] != {**B_SETTINGS['options'], 'refine_principal_point': True}:
            raise ValueError('Unexpected option change')
        try:
            branch_model(None, 'B', None)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-C branch accepted')
        print(json.dumps({'status': 'checked_without_BA', 'bound_files': len(manifest['bindings']),
                          'changed_option': manifest['changed_option'], 'initial_K': manifest['initial_K']}))
        return
    if args.action == '_worker':
        if base.read_json(OUTPUT / 'inputs.json') != manifest:
            raise ValueError('Child bindings or options differ from the parent')
        base.worker('C')
        return
    if OUTPUT.resolve() != OUTPUT:
        raise ValueError('Output path must not contain symlinks')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base.write(OUTPUT / 'inputs.json', manifest)
    case = OUTPUT / 'C'
    case.mkdir()
    command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin:/usr/sbin:/sbin',
               'TMPDIR=' + str(case), 'PYTHONDONTWRITEBYTECODE=1', str(base.RUNTIME / 'bin/python'),
               '-B', str(Path(__file__).resolve()), '_worker']
    base.write(case / 'command.json', {'argv': command, 'limits': base.LIMITS,
                                     'inputs_sha256': base.sha(OUTPUT / 'inputs.json')})
    status = base.supervise(command, case, timeout=base.LIMITS['timeout_seconds'], max_rss=base.LIMITS['max_rss_bytes'])
    status['final_log_hashes'] = {str(case / log): base.sha(case / log)
                                 for log in ('runtime.stdout', 'runtime.stderr', 'worker.log')
                                 if (case / log).is_file()}
    base.write(case / 'supervision.json', status)
    base.verify(manifest['bindings'])
    if status['returncode'] != 0 or status['reason'] is not None:
        raise RuntimeError('C failed; retain partial outputs and supervision logs')
    expected = {**B_SETTINGS, 'options': manifest['options']}
    if base.read_json(case / 'solver-settings.json') != expected:
        raise ValueError('Executed C settings/config differ from B beyond principal-point refinement')
    if base.read_json(case / 'result.json')['branch'] != 'C':
        raise ValueError('Incorrect result branch identity')
    base.write(OUTPUT / 'control-check.json', {'status': 'passed', 'control_B': str(CONTROL / 'B'),
        'changed_option': manifest['changed_option'], 'C_settings_sha256': base.sha(case / 'solver-settings.json')})
    print('C completed; original B retained as the control, with only principal-point refinement changed.')


if __name__ == '__main__':
    main()
