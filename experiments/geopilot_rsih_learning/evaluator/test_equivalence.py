"""Bounded equivalence and watchdog checks; never emits benchmark scores."""
import importlib.util
import json
import os
import laspy
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch, Mock
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'out/usegeo_mesh_benchmark/paper-readiness-20260918/release-v1/source/code'))
from usegeo_mesh_benchmark import benchmark as old
spec = importlib.util.spec_from_file_location('equal_v3', HERE / 'benchmark.py')
new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)


def main():
    new.runtime_versions()
    op, np_ = old.load_protocol(), new.load_protocol()
    np_['runtime']['implementation_id'] = op['runtime']['implementation_id']
    assert np_ == op
    # Large and small triangles, ties, far points, UTM coordinates, threshold edges.
    vertices = np.array([[0,0,0],[1000,0,0],[0,1000,0],
                         [1,1,1],[2,1,1],[1,2,1], [1,1,2],[1.01,1,2],[1,1.01,2]], dtype=float)
    triangles = np.array([[0,1,2],[3,4,5],[6,7,8],[3,4,5]])
    rng = np.random.default_rng(917)
    queries = np.concatenate([rng.uniform(-100,1100,(257,3)),
                             [[10,10,z] for z in [np.nextafter(.2,0),.2,np.nextafter(.2,1)]],
                             [[1e6,1e6,1e6],[1.2,1.2,1],[1,1,2]]])
    checks = []
    for shift in [np.zeros(3), np.array([500000,4400000,200])]:
        # Force reductions with many query candidates while retaining original safety limit in production.
        base_old, base_new = old.load_protocol, new.load_protocol
        def limited(base):
            p = base(); p['raycast']['max_materialized_candidate_pairs'] = 50; return p
        old.load_protocol = lambda: limited(base_old)
        new.load_protocol = lambda: limited(base_new)
        try:
            a = old.exact_triangle_queries(vertices+shift,triangles,queries+shift,np.zeros(3),chunk=100)
            b = new.exact_triangle_queries(vertices+shift,triangles,queries+shift,np.zeros(3),chunk=100)
            assert all(np.array_equal(x,y) for x,y in zip(a,b))
            assert np.array_equal(a[0] <= .2,b[0] <= .2)
        finally:
            old.load_protocol, new.load_protocol = base_old, base_new
        # Independent all-triangle oracle, stable lowest-id tie.
        faces=(vertices+shift)[triangles]
        brute=np.array([old._point_triangle_distances(np.broadcast_to(q, (len(faces),3)),faces) for q in queries+shift])
        assert np.array_equal(a[0], brute.min(axis=1))
        assert np.array_equal(a[1], brute.argmin(axis=1))
        checks.append({'shift':shift.tolist(),'queries':len(queries),'distance_and_id_exact':True})
    # Full small metric suite equality (same sample stream and reference arrays).
    full=rng.uniform(0,2,(1000,3)); refined=rng.uniform(0,2,(300,3))
    assert old.metrics_from_arrays(vertices,triangles,full,refined,sample_count=1000)[0] == new.metrics_from_arrays(vertices,triangles,full,refined,sample_count=1000)[0]
    # Single-query excessive candidates fail closed rather than exceeding allocation cap.
    base=new.load_protocol
    new.load_protocol=lambda: {**base(), 'raycast': {**base()['raycast'], 'max_materialized_candidate_pairs':1}}
    try:
        new.exact_triangle_queries(vertices,triangles,np.array([[1.2,1.2,1.]]),np.zeros(3))
        raise AssertionError('expected candidate limit')
    except new.EvaluatorFailure as exc:
        assert 'materialization limit' in str(exc)
    finally:
        new.load_protocol=base
    # Strict PLY validation remains byte-identical behavior for invalid/degenerate mesh.
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'bad.ply';p.write_bytes(old.PLY_HEADER.replace(b'{vertices}',b'3').replace(b'{triangles}',b'1') + np.zeros((3,3),dtype='<f8').tobytes() + b'\x03' + np.array([0,1,2],dtype='<i4').tobytes())
        for module in [old,new]:
            try: module.read_ply(p);raise AssertionError('accepted degenerate')
            except module.InvalidInput: pass
        for reason, timeout, limit in [('timeout',-.1,10**12),('RSS',10,1)]:
            proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                new._watch_process(proc,time.monotonic(),timeout,limit,.01,Path(tmp)/'cancel.json')
                raise AssertionError('expected watchdog')
            except new.EvaluatorFailure as exc:
                assert reason in str(exc),str(exc)
                assert exc.peak_rss_bytes is not None and exc.peak_rss_bytes>0
                assert proc.poll() is not None
    # Real child-process smoke on synthetic inputs, exercising work-order auth and progress.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); ref=root/'evaluator-only'/'Dataset-1'; ref.mkdir(parents=True)
        mesh=root/'mesh.ply'
        simple=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='<f8')
        mesh.write_bytes(old.PLY_HEADER.replace(b'{vertices}',b'3').replace(b'{triangles}',b'1') + simple.tobytes() + b'\x03' + np.array([0,1,2],dtype='<i4').tobytes())
        for name in ['full_lidar.las','refined_lidar.las']:
            las=laspy.LasData(laspy.LasHeader(point_format=3,version='1.2'))
            las.x=[0,.5,0];las.y=[0,0,.5];las.z=[0,0,0];las.write(ref/name)
        order={'token':'synthetic-test-only','parent_pid':os.getpid(),'bundle':str(root),
               'scene':'Dataset-1','mesh':str(mesh),'result':str(root/'worker-result.json'),
               'cancel':str(root/'cancel.json'),'progress':str(root/'progress.jsonl'),
               'deadline':time.monotonic()+30}
        (root/'order.json').write_text(json.dumps(order))
        env=dict(os.environ,USEGEO_MESH_WORKER_TOKEN=order['token'])
        done=subprocess.run([sys.executable,str(HERE/'benchmark.py'),'score-worker','--work-order',str(root/'order.json')],env=env,capture_output=True,text=True,timeout=30)
        assert done.returncode==0,done.stderr
        result=json.loads((root/'worker-result.json').read_text())
        assert result['counts']['surface_samples']==1000000
        assert 'worker_complete' in (root/'progress.jsonl').read_text()
    # Failure publication retains already-known bindings and an observed RSS.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); output=root/'failure'
        known={'mesh_sha256':'a'*64,'submission_contract_sha256':'b'*64,
               'protocol_sha256':new.sha256_file(new.PROTOCOL_PATH),
               'runtime_implementation_id':new.load_protocol()['runtime']['implementation_id']}
        failure=new.EvaluatorFailure('worker scene timeout exceeded');failure.peak_rss_bytes=123456
        new.platform.platform()  # cache platform's own subprocess before the test double
        process=Mock();process.poll.return_value=1
        with patch.object(new,'verify_bundle'), patch.object(new,'snapshot_submission',return_value={}), \
             patch.object(new,'read_ply',return_value=(vertices,triangles)), \
             patch.object(new,'_bindings',return_value=known), \
             patch.object(new.subprocess,'Popen',return_value=process), \
             patch.object(new,'_watch_process',side_effect=failure):
            code,score=new.score_submission(root,root/'lock','Dataset-1',root/'submission',output)
        assert code==1 and score['status']=='error' and score['bindings']==known
        run=json.loads((output/'run_manifest.json').read_text())
        assert run['peak_worker_rss_bytes']==123456 and run['bindings']==known
        assert set(p.name for p in output.iterdir())=={'score.json','run_manifest.json'}
        assert output.with_name('failure.progress.jsonl').is_file()
    print(json.dumps({'status':'pass','checks':checks,'metrics_equal':True,'candidate_cap':True,'invalid_geometry':True,'timeout_and_rss':True},indent=2))

if __name__=='__main__': main()
