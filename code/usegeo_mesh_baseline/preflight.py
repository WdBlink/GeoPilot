"""Check pinned baseline APIs on synthetic points only; no dataset access."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pycolmap
import scipy
from scipy.spatial import Delaunay
from runner import CONFIG, options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert platform.python_version() == '3.10.20'
    assert (pycolmap.__version__, np.__version__, scipy.__version__) == ('3.12.6', '1.26.4', '1.11.1')
    pycolmap.set_random_seed(917)
    opts=options()
    extraction,matching,reader,mapper=(opts[n] for n in ['extraction','matching','reader','mapper'])
    mapper.mapper.num_threads = 1
    points = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
    target = 2.5*points + [498000.,4379500.,120.]
    transform = pycolmap.estimate_sim3d(points, target)
    assert transform is not None
    error = float(np.max(np.abs(transform * points-target)))
    assert error < 1e-8
    triangulation = Delaunay(points[[0,1,2,4],:2])
    assert triangulation.simplices.shape == (2,3)
    APIs = ['extract_features','match_sequential','match_spatial','incremental_mapping','estimate_sim3d','set_random_seed']
    assert all(callable(getattr(pycolmap, name)) for name in APIs)
    here = Path(__file__).resolve().parent
    result = {'status':'API_PREFLIGHT_PASS_NOT_RECONSTRUCTION',
              'versions':{'python':platform.python_version(),'pycolmap':pycolmap.__version__,
                          'numpy':np.__version__,'scipy':scipy.__version__},
              'has_cuda':pycolmap.has_cuda,'device':'cpu','camera_mode':str(pycolmap.CameraMode.PER_IMAGE),'config':CONFIG,
              'dataset_files_read':0,'seed':917,'sim3_synthetic_max_error_m':error,
              'synthetic_triangles':len(triangulation.simplices),
              'preregistration_sha256':hashlib.sha256((here/'PREREGISTRATION.md').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'api_signatures':{name:getattr(pycolmap,name).__doc__ for name in APIs},
              'options':{'reader':reader.todict(),'extraction':extraction.todict(),'matching':matching.todict(),
                         'sequential':opts['sequential'].todict(),'spatial':opts['spatial'].todict(),
                         'verification':pycolmap.TwoViewGeometryOptions().todict(),'mapper':mapper.todict()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,default=str,allow_nan=False)
    print(result['status'])


if __name__ == '__main__':
    main()
