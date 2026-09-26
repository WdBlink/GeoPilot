"""No reconstruction: check the parameter boundary and independent prefix layout."""
import json
from pathlib import Path
import tempfile
import unittest

from fixed_prefix_control import CASES, make_request, sha, stage_prefix, verify


class FixedPrefixControlTest(unittest.TestCase):
    def test_cases_and_independent_rebased_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            history, case = root / 'history', root / 'A'
            case.mkdir()
            workspace = history / 'nodes/prepare'
            names = {'scene': 'nodes/prepare/scene.mvs', 'dense': 'nodes/densify/densify.mvs',
                     'dense_cloud': 'nodes/densify/densify.ply',
                     'depth_manifest': 'nodes/densify/depth-manifest.json',
                     'coverage_prepare_path': 'nodes/prepare/coverage-prepare.json',
                     'coverage_path': 'nodes/densify/coverage-dense.json'}
            files = list(names.values()) + ['nodes/prepare/undistorted/images/image.jpg',
                                          'nodes/prepare/depth0000.dmap', 'nodes/prepare/sfm/inputs.json']
            for name in files:
                path = history / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixed test input')
            manifest = history / names['depth_manifest']
            depth = str(workspace / 'depth0000.dmap')
            manifest.write_text(json.dumps({'depth0000.dmap': {'path': depth, 'sha256': sha(depth)}}))
            parent = {key: str(history / name) for key, name in names.items()}
            parent.update(mvs_workspace=str(workspace), stage='dense', transform=[[1, 0, 0, 0]],
                          sfm_evidence_files=[str(workspace / 'sfm/inputs.json')],
                          undistorted_files=[str(workspace / 'undistorted/images/image.jpg')],
                          artifact_hashes={str(history / name): sha(history / name) for name in files})
            before = json.dumps(parent, sort_keys=True)
            staged = stage_prefix(parent, case, history)
            self.assertEqual(json.dumps(parent, sort_keys=True), before)
            verify(parent['artifact_hashes'])
            verify(staged['artifact_hashes'])
            self.assertNotEqual(Path(parent['dense']).stat().st_ino, Path(staged['dense']).stat().st_ino)
            self.assertTrue((Path(staged['mvs_workspace']) / 'undistorted/images/image.jpg').is_file())
            copied_manifest = json.loads(Path(staged['depth_manifest']).read_text())
            self.assertTrue(Path(copied_manifest['depth0000.dmap']['path']).is_relative_to(case))
            for label, value in [('A', .25), ('A-repeat', .25), ('B', .5)]:
                request = make_request(label, staged, case, 'fixed-input')
                self.assertEqual(request['node']['action'], 'mesh')
                self.assertEqual(request['node']['parameters'], {'decimate': value})
            self.assertEqual(len(CASES), 3)
            with self.assertRaises(ValueError):
                make_request('0.75', staged, case, 'fixed-input')
            with self.assertRaises(ValueError):
                verify({staged['dense']: '0' * 64})
            # The native tool may add a log to its isolated cwd, never history.
            (Path(staged['mvs_workspace']) / 'native.log').write_text('test')
            self.assertFalse((workspace / 'native.log').exists())


if __name__ == '__main__':
    unittest.main()
