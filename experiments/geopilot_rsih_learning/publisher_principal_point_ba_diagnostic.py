"""Exploratory D: shared-camera BA with the publisher's fixed principal point."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import shared_camera_ba_diagnostic as base

SOURCE_SHA = '07d85983a366f8b928e104a47884d9ddaea343554efd3b0c0b0210db9c5f71ac'
CONTROL = base.OUTPUT
OUTPUT = base.BASE / 'publisher-principal-point-ba-run'
PLAN = base.BASE / 'publisher-principal-point-ba-plan.md'
COMMENT = base.BASE / 'publisher-camera-comment-2243013934.json'
CONVENTIONS = base.BASE / 'publisher-camera-conventions.md'
TABLE = base.INPUTS / 'Image_orientations_dataset1.xyz'
D_K = [4639.021835367615, 3970.288, 2601.560, 0.0]
CONTROL_SHA = {
    'inputs.json': 'd6a292e04684544845919e880f18625a3726e4290f156645eddb3361dc53682c',
    'B/result.json': 'b5750c704304ef47df1d9151a9be7f5a397ef8360441f737eafdb862c42d6c24',
    'B/solver-settings.json': '7d180581a72b8eded93779d8e920493347e4a40260b82c2f5bfa104636a1a535',
    'B/supervision.json': '02835d6e49ea0ea9017460dca3e361fb02256429c2a810104bd2ef2e08f787ba',
}
EVIDENCE_SHA = {
    str(COMMENT): '5d2eed6452302ae9d6980c77db0b81d51183172e13d84329b12585c60c981dfa',
    str(TABLE): '523ac45c41e097b7396a66239c1be932f4c0e05c6e354667cfcfa5d57a783b8b',
}
base.verify({str(Path(base.__file__).resolve()): SOURCE_SHA,
             **{str(CONTROL / name): digest for name, digest in CONTROL_SHA.items()}})
B_SETTINGS = base.read_json(CONTROL / 'B/solver-settings.json')
ORIGINAL_INITIAL, ORIGINAL_BRANCH = base.initial_cameras, base.branch_model


def json_value(value):
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def publisher_evidence():
    """Validate only the specific documented PP mapping, never OPK/c/distortion."""
    base.verify(EVIDENCE_SHA)
    comment = base.read_json(COMMENT)['comment']
    if (comment['databaseId'] != 2243013934 or comment['author']['login'] != '3DOM-FBK'
            or comment['authorAssociation'] != 'OWNER'
            or 'upper left corner' not in comment['body']
            or '(3970.288, 2601.560)' not in comment['body']):
        raise ValueError('Publisher principal-point evidence identity mismatch')
    rows = [line.split() for line in TABLE.read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    names = {r['image'] for r in base.read_json(CONTROL / 'B/before-ba.json')['center_residuals_m']}
    if (len(rows) != 224 or any(len(row) != 15 for row in rows)
            or {row[0] for row in rows} != names
            or {(row[8], row[9]) for row in rows} != {('3970.288', '-2601.560')}
            or [float(rows[0][8]), -float(rows[0][9])] != D_K[1:3]):
        raise ValueError('All 224 table principal points must match the documented mapping')
    return {'comment': str(COMMENT), 'comment_url': comment['url'], 'table': str(TABLE),
            'rows': 224, 'table_x0_y0': [3970.288, -2601.560],
            'mapping': 'cx=x0; cy=-y0 for this fixed negative-y0 table only',
            'principal_point': D_K[1:3]}


def initial_cameras(path, model):
    initial = ORIGINAL_INITIAL(path, model)  # Validate all 224 original EXIF K entries first.
    initial[1:3] = D_K[1:3]
    if initial.tolist() != D_K:
        raise ValueError('D may change only the two initial principal-point coordinates')
    config = base.pc.BundleAdjustmentConfig()
    for image_id in sorted(model.reg_image_ids()):
        config.add_image(image_id)
    config.fix_gauge(base.pc.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    if json_value(config.todict()) != B_SETTINGS['config']:
        raise ValueError('D image selection/gauge config differs from B')
    return initial


def branch_model(source, name, initial):
    if name != 'D' or initial.tolist() != D_K:
        raise ValueError('Only the fixed D branch and publisher PP initialization are permitted')
    return ORIGINAL_BRANCH(source, 'B', initial)


# The child imports this same file and applies the same three local overrides.
base.OUTPUT, base.initial_cameras, base.branch_model = OUTPUT, initial_cameras, branch_model


def inputs():
    if Path(sys.prefix).resolve() != base.RUNTIME.resolve() or base.pc.__version__ != '3.12.6':
        raise RuntimeError('Use the fixed baseline-runtime Python with pycolmap 3.12.6')
    prior = base.read_json(CONTROL / 'inputs.json')
    result = base.read_json(CONTROL / 'B/result.json')
    supervision = base.read_json(CONTROL / 'B/supervision.json')
    if (result['status'] != 'completed' or result['branch'] != 'B' or not result['solution_usable']
            or supervision['returncode'] != 0 or supervision['reason'] is not None
            or prior['limits'] != base.LIMITS or prior['initial_K'] != base.INITIAL_K
            or prior['options'] != B_SETTINGS['options']
            or json_value(base.options().todict()) != B_SETTINGS['options']
            or B_SETTINGS['options']['refine_principal_point'] is not False
            or B_SETTINGS['options']['refine_focal_length'] is not True
            or B_SETTINGS['options']['refine_extra_params'] is not True):
        raise ValueError('B control, initialization or unchanged solver settings are invalid')
    bound = {**prior['bindings'], **result['copied_input_hashes'], **result['output_hashes'],
             **supervision['final_log_hashes'], **EVIDENCE_SHA,
             **{str(CONTROL / name): digest for name, digest in CONTROL_SHA.items()}}
    bound[str(Path(base.__file__).resolve())] = SOURCE_SHA
    for path in (Path(__file__).resolve(), PLAN, CONVENTIONS):
        bound[str(path)] = base.sha(path)
    base.verify(bound)
    before = base.read_json(CONTROL / 'B/before-ba.json')
    if len(before['cameras']) != 1 or before['cameras'][0]['params'] != base.INITIAL_K:
        raise ValueError('B initial K does not match the fixed control')
    if [i for i, (a, b) in enumerate(zip(base.INITIAL_K, D_K)) if a != b] != [1, 2]:
        raise ValueError('D must change only principal-point coordinates')
    return {**prior, 'bindings': bound, 'initial_K': D_K, 'branch': 'D',
            'exploratory_after_observed_results': True, 'formal_candidate': False,
            'control_B': str(CONTROL / 'B'), 'control_initial_K': base.INITIAL_K,
            'changed_initial_parameters': {'cx': [3976.0, D_K[1]], 'cy': [2652.0, D_K[2]]},
            'publisher_principal_point': publisher_evidence()}


def main():
    if not __debug__:
        raise RuntimeError('Optimized Python is unsupported')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', '_worker'))
    args = parser.parse_args()
    manifest = inputs()
    if args.action == 'check':
        model = base.pc.Reconstruction(str(base.MODEL))
        if initial_cameras(base.SFM / 'cameras.json', model).tolist() != manifest['initial_K']:
            raise ValueError('Actual D initialization differs from the manifest')
        for name, initial in [('B', base.np.array(D_K)), ('D', base.np.array(base.INITIAL_K))]:
            try:
                branch_model(None, name, initial)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid branch or initial K accepted')
        base.verify(manifest['bindings'])
        print(json.dumps({'status': 'checked_without_BA', 'bound_files': len(manifest['bindings']),
                          'branch': 'D', 'initial_K': manifest['initial_K'], 'B_settings_unchanged': True}))
        return
    if args.action == '_worker':
        if base.read_json(OUTPUT / 'inputs.json') != manifest:
            raise ValueError('Child bindings/initialization differ from the parent')
        base.worker('D')
        return
    if OUTPUT.resolve() != OUTPUT:
        raise ValueError('Output path must not contain symlinks')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    base.write(OUTPUT / 'inputs.json', manifest)
    case = OUTPUT / 'D'
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
        raise RuntimeError('D failed; retain partial outputs and supervision logs')
    before, after = [base.read_json(case / name) for name in ('before-ba.json', 'after-ba.json')]
    if (base.read_json(case / 'solver-settings.json') != B_SETTINGS
            or base.read_json(case / 'result.json')['branch'] != 'D'
            or len(before['cameras']) != 1 or before['cameras'][0]['params'] != D_K
            or len(after['cameras']) != 1 or after['cameras'][0]['params'][1:3] != D_K[1:3]):
        raise ValueError('Executed D settings/config, identity or fixed PP differs from the declaration')
    base.write(OUTPUT / 'control-check.json', {'status': 'passed', 'control_B': str(CONTROL / 'B'),
        'changed_initial_parameters': manifest['changed_initial_parameters'],
        'actual_initial_K': before['cameras'][0]['params'], 'options_config_identical_to_B': True,
        'D_settings_sha256': base.sha(case / 'solver-settings.json')})
    print('D completed; original B retained, with only the two fixed principal-point coordinates changed.')


if __name__ == '__main__':
    main()
