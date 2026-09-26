"""Compare exact distances on bounded r3 reference windows; no formal score."""
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path
import laspy
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1/source/code'))
from usegeo_mesh_benchmark import benchmark as old
spec=importlib.util.spec_from_file_location('equal_v3',HERE/'benchmark.py')
new=importlib.util.module_from_spec(spec);spec.loader.exec_module(new)
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('180 second diagnostic cap')))
signal.alarm(180)
mesh=ROOT/'out/geopilot-rsih-p0-mu9geojt-r3/submission/mesh.ply'
v,f=old.read_ply(mesh)
with laspy.open(ROOT/'out/usegeo_benchmark/prepared/v1/evaluator-only/Dataset-1/refined_lidar.las') as reader:
    count=reader.header.point_count
    starts=np.linspace(0,count-1000,100,dtype=int).tolist()
    blocks=[]
    for start in starts:
        reader.seek(start);p=reader.read_points(1000);blocks.append(np.column_stack((p.x,p.y,p.z)))
q=np.concatenate(blocks);results=[];values=[]
for module in [old,new]:
    begin=time.perf_counter();value=module.exact_triangle_queries(v,f,q,np.zeros(3))
    values.append(value);results.append({'runtime':module.load_protocol()['runtime']['implementation_id'],'seconds':time.perf_counter()-begin})
assert all(np.array_equal(a,b) for a,b in zip(*values))
print(json.dumps({'status':'pass','formal_score':False,'mesh_sha256':old.sha256_file(mesh),'queries':len(q),'total_reference':count,'sampling':'100 contiguous windows of 1000 at evenly spaced file offsets (not spatial-uniform)','window_starts':starts,'distance_bitwise_equal':True,'primitive_ids_equal':True,'threshold_mask_equal':bool(np.array_equal(values[0][0]<=.2,values[1][0]<=.2)),'timings':results},indent=2))
