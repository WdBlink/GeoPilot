"""Fresh F-U/S-U camera-to-mesh development controls; never select or refit a sparse model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
HELPER = ENTRY.with_name('publisher_raster_full_sparse.py')
spec = importlib.util.spec_from_file_location('publisher_full_sparse', HELPER)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
ba, fixed, pc, np = full.ba, full.fixed, full.pc, full.np
import tools
SOURCE = full.OUTPUT
OUTPUT = ba.BASE/'publisher-raster-mesh-run'
PLAN = ba.BASE/'publisher-raster-mesh-plan.md'
CASES = ('F-U', 'S-U')
PARAMETERS = {'undistort_max_image_size': 2400, 'densify': {'resolution-level': 1}, 'mesh': {'decimate': .5},
    'undistort_call': {'output_type':'COLMAP', 'copy_policy':'copy', 'num_patch_match_src_images':20},
    'transform_application': 'once at native OBJ to world PLY export; never transform source model',
    'export_atol_m': 1e-9, 'export_rtol': 0.0, 'pair_order': list(CASES)}
LIMITS = {'timeout_seconds':1800, 'max_rss_bytes':32*1024**3,
          'min_free_bytes':60*1024**3, 'openmvs_threads':8}


def undistort_options():
    return pc.UndistortCameraOptions(max_image_size=2400)


def transform_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3,4) or not np.isfinite(matrix).all():
        raise ValueError('A finite recorded 3x4 world transform is required')
    singular = np.linalg.svd(matrix[:,:3], compute_uv=False)
    if singular.min() <= 0 or np.linalg.det(matrix[:,:3]) <= 0 or not np.allclose(singular, singular[0], atol=1e-12, rtol=1e-10):
        raise ValueError('Recorded transform is not a proper positive-scale Sim3')
    return matrix


def model_identity(model, names):
    """Native semantic identity includes unregistered images without requiring their poses."""
    digest = hashlib.sha256()
    def add(value):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n')
    registered = set(model.reg_image_ids())
    if not registered or not {i.name for i in model.images.values()}.issubset(names):
        raise ValueError('No registered images or unexpected sparse image')
    for ident, camera in sorted(model.cameras.items()):
        if camera.model != pc.CameraModelId.SIMPLE_RADIAL or [camera.width,camera.height] != full.DIMENSIONS:
            raise ValueError('Wrong source U raster/model')
        add([ident,str(camera.model),camera.width,camera.height,camera.params.tolist()])
    for ident, image in sorted(model.images.items()):
        add([ident,image.name,image.camera_id,image.frame_id,ident in registered])
        if ident in registered:
            add(image.cam_from_world().matrix().tolist())
        digest.update(np.asarray([p.xy for p in image.points2D], dtype='<f8').tobytes())
        digest.update(np.asarray([p.point3D_id for p in image.points2D], dtype='<u8').tobytes())
    for ident, point in sorted(model.points3D.items()):
        add([ident,point.xyz.tolist(),point.color.tolist(),sorted((p.image_id,p.point2D_idx) for p in point.track.elements)])
    return {'sha256':digest.hexdigest(), 'registered_names':sorted(model.images[i].name for i in registered),
        'points3D':model.num_points3D(), 'observations':model.compute_num_observations()}


def verify_export(path, vertices, faces, matrix):
    """Check the bound binary writer's PLY is exactly the OBJ with one recorded transform."""
    header = ba.legacy.sparse.HEADER.replace(b'{v}',str(len(vertices)).encode()).replace(b'{f}',str(len(faces)).encode())
    with Path(path).open('rb') as stream:
        if stream.read(len(header)) != header:
            raise ValueError('Unexpected exported PLY layout')
    if Path(path).stat().st_size != len(header)+len(vertices)*24+len(faces)*13:
        raise ValueError('Exported PLY length mismatch')
    v = np.memmap(path,dtype='<f8',mode='r',offset=len(header),shape=(len(vertices),3))
    f = np.memmap(path,dtype=[('count','u1'),('indices','<i4',(3,))],mode='r',
                  offset=len(header)+len(vertices)*24,shape=(len(faces),))
    maximum = 0.
    for start in range(0,len(vertices),100000):
        expected = vertices[start:start+100000] @ matrix[:,:3].T + matrix[:,3]
        actual = v[start:start+100000]
        if not np.isfinite(actual).all(): raise ValueError('Nonfinite world vertices')
        maximum = max(maximum,float(np.max(np.abs(actual-expected))))
    if maximum > PARAMETERS['export_atol_m'] or not np.all(f['count']==3) or not np.array_equal(f['indices'],faces):
        raise ValueError('Mesh export is not the same topology with exactly one recorded transform')
    return {'passed':True,'vertices':len(vertices),'faces':len(faces),'max_abs_vertex_difference_m':maximum}


def self_test():
    full.tri.self_test()
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int32)
    matrix = transform_matrix([[0.,-2.,0.,500000.],[2.,0.,0.,4400000.],[0.,0.,2.,200.]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'toy.ply'
        ba.legacy.sparse.write_mesh(path,vertices @ matrix[:,:3].T+matrix[:,3],faces)
        assert verify_export(path,vertices,faces,matrix)['passed']
        wrong = matrix.copy(); wrong[0,3] += 1.
        try: verify_export(path,vertices,faces,wrong)
        except ValueError: pass
        else: raise AssertionError('Incorrect export transform accepted')
    assert ba.legacy.THREADS == 8
    assert fixed.plain(undistort_options().todict())['max_image_size'] == 2400
    return 'PASS: inherited camera checks, proper Sim3, once-only binary mesh export and wrong-transform rejection'


def inputs():
    pins = {str(HELPER):'f97d37f220a3c02245955e9c7cce091bc61c4642288f863a71d22da5a9e1fe76',
        str(full.HELPER):'4872c4bf0e27284d0c1cc54a4f785d22f1ba4353eede160d13b57e725bf554a4',
        str(full.U/'input_manifest.json'):'0327334e7607a77ab81fee4ffd6dcca4a90d76a6d45221956f78307b3ebd4ecc'}
    ba.verify(pins)
    prior = ba.read_json(ba.BASE/'publisher-raster-pair-run/inputs.json')
    launch = ba.read_json(ba.SOURCE/'launch.json')
    common = {**prior['worker_common_bindings'], **pins}
    # Input manifest contains no per-image OPK; worker receives only its recorded model/transform.
    for path in (ENTRY, PLAN, full.PLAN, full.tri.HELPER): common[str(path)] = ba.sha(path)
    for name in ('InterfaceCOLMAP','DensifyPointCloud','ReconstructMesh','RefineMesh'):
        path = str(ba.legacy.TOOLS/name); common[path] = launch['bindings'][path]
    for name in ('densify-help.txt','mesh-help.txt'):
        path = str(ba.ROOT/'out/geopilot-learning-20260921'/name); common[path] = launch['bindings'][path]
    ba.verify(common)
    manifest = {'schema':'publisher-raster-mesh/1','formal_candidate':False,'parameters':PARAMETERS,'limits':LIMITS,
        'runtime':prior['runtime'],'worker_common_bindings':common,'bindings':dict(common),
        'undistort_options':fixed.plain(undistort_options().todict()),'free_bytes':shutil.disk_usage(ba.BASE).free,
        'implicit_defaults':'Same bound OpenMVS2.4.0 binary/help; no .cfg in fresh workspace; other defaults remain implicit'}
    required = [SOURCE/'inputs.json',SOURCE/'result.json',*[SOURCE/n/'result.json' for n in ('features','matches',*CASES)]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return {**manifest,'readiness':'not_ready','reasons':['Full U224 sparse campaign is incomplete'],'missing':missing}
    source, result = ba.read_json(SOURCE/'inputs.json'),ba.read_json(SOURCE/'result.json')
    if result['status'] != 'completed':
        return {**manifest,'readiness':'not_ready','reasons':['Full sparse campaign not completed']}
    ba.verify(source['bindings'])
    names, images = source['names'],{Path(p).name:h for p,h in source['images'].items()}
    if len(names)!=224 or set(names)!=set(images) or source['parameters']!=full.PARAMETERS:
        raise ValueError('U224 sparse protocol/input identity drift')
    bound = manifest['bindings']
    bound.update(source['bindings'])
    for path in required: bound[str(path)] = ba.sha(path)
    for stage in ('features','matches',*CASES):
        path = SOURCE/stage/'result.json'
        if ba.sha(path)!=result['stages'][stage]['sha256'] or ba.read_json(path)['status']!='completed':
            raise ValueError('Full sparse completion chain changed')
    feature = ba.read_json(SOURCE/'features/result.json')
    counts_path = SOURCE/'features/feature-counts.json'
    bound[str(counts_path)] = feature['output_hashes'][str(counts_path)]
    feature_counts = ba.read_json(counts_path)
    if set(feature_counts)!=set(names): raise ValueError('Full feature input denominator changed')
    sources, reasons = {},[]
    for arm in CASES:
        record = ba.read_json(SOURCE/arm/'result.json')
        chosen = record['selected_model']
        if chosen is None or chosen['transform'] is None:
            reasons.append(arm+': missing preselected model or recorded transform'); continue
        if str(chosen['id']) != record['model_ranking'][0]: raise ValueError('Selected model ranking mismatch')
        entry = record['all_models'][str(chosen['id'])]
        if chosen['path']!=entry['path']: raise ValueError('Selected model path mismatch')
        model = Path(chosen['path'])
        if not model.resolve().is_relative_to(SOURCE/arm) or model.is_symlink(): raise ValueError('Unsafe selected model path')
        files = {str(model/f):entry['output_hashes'][str(model/f)] for f in ba.MODEL_SHA}
        ba.verify(files)
        report_path = Path(entry['report_path'])
        bound[str(report_path)] = record['output_hashes'][str(report_path)]
        report = ba.read_json(report_path)
        matrix = transform_matrix(chosen['transform'])
        if arm=='F-U':
            expected = np.asarray(record['transform']['local_to_world'])
            if not np.array_equal(matrix,expected) or not np.array_equal(matrix[:,:3],np.eye(3)):
                raise ValueError('F-U must retain the unique C0 translation')
        elif not np.array_equal(matrix,np.asarray(report['allowed_XYZ_alignment']['matrix'])):
            raise ValueError('S-U must retain its recorded XYZ Sim3')
        if {r['name'] for r in report['images']}!=set(names): raise ValueError('Selected report omits input images')
        if report['points3D']<=0:
            reasons.append(arm+': selected model has zero points'); continue
        bound.update(files)
        sources[arm] = {'model':str(model),'model_hashes':files,'transform':matrix.tolist(),
            'selected_model':chosen,'sparse_report':report,'source_result_sha256':ba.sha(SOURCE/arm/'result.json'),
            'source_report_sha256':bound[str(report_path)]}
    ba.verify(bound)
    if reasons: return {**manifest,'readiness':'not_eligible','reasons':reasons}
    bound.update(source['images'])
    manifest.update(readiness='ready',sources=sources,names=names,images=images,image_bindings=source['images'],
        input_manifest_sha256=pins[str(full.U/'input_manifest.json')],feature_counts=feature_counts,
        candidate_pairs=source['pairs'],source_inputs_sha256=ba.sha(SOURCE/'inputs.json'))
    return manifest


def coverage(manifest, source, files):
    support = {r['name']:r for r in source['sparse_report']['images']}
    candidates = {n for pair in manifest['candidate_pairs']['union'] for n in pair}
    records = {}
    for name in manifest['names']:
        item = support[name]
        registered = item['registered']
        omissions = ([] if registered else ['not_registered_in_selected_model'])
        if not item['point_observations']: omissions.append('no_sparse_point_observations')
        if name not in files: omissions.append('not_undistorted')
        records[name] = {'input_sha256':manifest['images'][name],'sfm_ingested':True,'matching_candidate':name in candidates,
            'registered':registered,'sparse_point_observations':item['point_observations'],
            'undistorted_sha256':ba.sha(files[name]) if name in files else None,'depth_maps':[],'omissions':omissions}
    return {'policy':'manual-U224-selected-sparse-mesh','input_manifest_sha256':manifest['input_manifest_sha256'],
        'input_count':224,'sfm_input_count':224,'matching_candidate_count':len(candidates),
        'registered_count':sum(r['registered'] for r in records.values()),'undistorted_count':len(files),
        'depth_image_count':0,'depth_map_count':0,'images':records,
        'sparse_evidence':'Reused bound fresh U224 features/matching and selected sparse model; no new SfM in mesh stage'}


def worker(case, m, check_only):
    ba.verify(m['worker_bindings'])
    if m['parameters']!=PARAMETERS or m['limits']!=LIMITS or m['undistort_options']!=fixed.plain(undistort_options().todict()):
        raise ValueError('Mesh parameters/runtime defaults drift')
    proof = self_test()
    if m['readiness']!='ready':
        ba.write(case/'worker-result.json',{'status':'checked_parameters_source_not_ready','readiness':m['readiness'],'self_test':proof})
        return
    source = m['source']
    for path in m['denied_inputs']:
        try:
            with Path(path).open('rb'): raise ValueError('Another arm or publisher table became readable')
        except PermissionError: pass
    copied, model_dir = {},case/'input-model'
    model_dir.mkdir()
    for original,digest in source['model_hashes'].items():
        target = model_dir/Path(original).name; shutil.copyfile(original,target); target.chmod(0o444); copied[str(target)] = digest
    ba.verify(copied)
    model = pc.Reconstruction(str(model_dir))
    identity = model_identity(model,set(m['names']))
    expected = {r['name'] for r in source['sparse_report']['images'] if r['registered']}
    if set(identity['registered_names'])!=expected or identity['points3D']!=source['sparse_report']['points3D']:
        raise ValueError('Selected sparse model differs from bound support report')
    roundtrip = case/'roundtrip-model'; roundtrip.mkdir(); model.write_binary(str(roundtrip))
    if model_identity(pc.Reconstruction(str(roundtrip)),set(m['names']))!=identity:
        raise ValueError('Native selected-model roundtrip changed geometry/observations/cameras')
    ba.write(case/'source-model-check.json',identity)
    if check_only:
        ba.verify(m['worker_bindings']); ba.verify(copied)
        ba.write(case/'worker-result.json',{'status':'checked_without_undistort_or_MVS','branch':m['arm'],
            'self_test':proof,'identity':identity,'copied_input_hashes':copied}); return
    prepare = case/'prepare'; prepare.mkdir()
    undistorted,scene = prepare/'undistorted',prepare/'scene.mvs'
    start = time.monotonic()
    ba.write(prepare/'undistort-command.json',{'input_model':str(model_dir),'input_images':str(full.U/'images'),
        'output':str(undistorted),'options':m['undistort_options'],**PARAMETERS['undistort_call']})
    pc.undistort_images(str(undistorted),str(model_dir),str(full.U/'images'),output_type='COLMAP',copy_policy=pc.CopyType.copy,
        num_patch_match_src_images=20,undistort_options=undistort_options())
    files = {p.name:p for p in (undistorted/'images').iterdir()}
    if not files or not set(files).issubset(expected) or any(not p.is_file() or p.is_symlink() for p in files.values()):
        raise ValueError('Invalid fresh undistorted image set')
    cov = coverage(m,source,files); cov_path=prepare/'coverage-prepare.json'; ba.write(cov_path,cov)
    evidence = case/'sparse-evidence.json'
    ba.write(evidence,{'source_inputs_sha256':m['source_inputs_sha256'],'source_result_sha256':source['source_result_sha256'],
        'source_report_sha256':source['source_report_sha256'],'selected_model':source['selected_model'],
        'feature_counts':m['feature_counts'],'candidate_pairs':m['candidate_pairs']})
    original_invoke = tools.invoke
    def invoke(argv,output,workdir=None):
        with (case/'commands.jsonl').open('a') as stream:
            stream.write(json.dumps({'argv':argv,'cwd':str(workdir or output),'tool_sha256':ba.sha(argv[0])})+'\n')
        original_invoke(argv,output,workdir)
    tools.invoke = invoke
    invoke([str(ba.legacy.TOOLS/'InterfaceCOLMAP'),'-i',str(undistorted),'-o',str(scene),
        '--image-folder',str(undistorted/'images'),'--archive-type','2','--max-threads','8'],prepare,prepare)
    with scene.open('rb') as stream:
        if stream.read(12)!=bytes.fromhex('4d5653490700000000000000'): raise ValueError('Unexpected fresh MVSI header')
        metadata = stream.read(16*1024**2)
    if any(('undistorted/images/'+n).encode() not in metadata for n in files): raise ValueError('MVS relative image layout changed')
    state = {'stage':'sparse','diagnostics':{},'scene':str(scene),'mesh':None,'transform':source['transform'],
        'mvs_workspace':str(prepare),'scene_id':'Dataset-1','track':'manual-publisher-undistorted',
        'input_manifest_sha256':m['input_manifest_sha256'],'input_image_count':224,'registered_image_count':len(expected),
        'undistorted_image_count':len(files),'coverage_prepare_path':str(cov_path),'coverage_path':str(cov_path),
        'undistorted_files':[str(files[n]) for n in sorted(files)],'sfm_evidence_files':[str(evidence)],'artifact_hashes':{str(scene):ba.sha(scene)}}
    ba.write(prepare/'state.json',state)
    timing = {'undistort_and_interface_seconds':time.monotonic()-start}
    for action in ('densify','mesh'):
        if any(prepare.glob('*.cfg')): raise ValueError('Unexpected OpenMVS configuration file')
        out = case/action; out.mkdir()
        request = {'scope':'development','inputs':str(full.U),'output':str(out),
            'node':{'id':action,'kind':'tool','action':action,'parameters':PARAMETERS[action]},'parent':state}
        ba.write(out/'request.json',request); start=time.monotonic()
        state = tools.real_action(request); ba.write(out/'state.json',state)
        timing[action+'_seconds'] = time.monotonic()-start
    vertices,faces=ba.legacy.read_obj(Path(state['native_obj']))
    if not len(vertices) or not len(faces): raise ValueError('No valid output mesh')
    export = verify_export(state['mesh']['path'],vertices,faces,transform_matrix(source['transform']))
    ba.write(case/'export-verification.json',export)
    final_coverage=ba.read_json(state['coverage_path'])
    ba.verify(m['worker_bindings']);ba.verify(copied);ba.verify(state['artifact_hashes'])
    ba.verify({state['mesh']['path']:state['mesh']['sha256']})
    ba.write(case/'worker-result.json',{'status':'completed','branch':m['arm'],'formal_candidate':False,
        'mesh_quality_evaluated':False,'mesh':state['mesh'],'copied_input_hashes':copied,'phase_seconds':timing,
        'source_selected_model':source['selected_model'],'coverage_path':state['coverage_path'],
        'full_224_coverage':all(not x['omissions'] and len(x['depth_maps'])==1 for x in final_coverage['images'].values()),
        'export_verification':export})


def main():
    if not __debug__ or Path(sys.prefix).resolve()!=ba.RUNTIME.resolve() or pc.__version__!='3.12.6' or ba.legacy.THREADS!=8:
        raise RuntimeError('Use fixed baseline-runtime -B / OpenMVS8, never -O')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('self-test','check','run','_check_worker','_worker'));parser.add_argument('manifest',nargs='?')
    args=parser.parse_args()
    if args.action=='self-test': print(self_test());return
    if args.action.startswith('_'):
        path=Path(args.manifest).resolve();worker(path.parent,ba.read_json(path),args.action=='_check_worker');return
    campaign=inputs()
    if OUTPUT.exists(): raise ValueError('Fresh output required')
    if args.action=='run' and campaign['readiness']!='ready': raise RuntimeError('Source not eligible: '+str(campaign.get('reasons')))
    root=Path(tempfile.mkdtemp(prefix='publisher-raster-mesh-check-')).resolve() if args.action=='check' else OUTPUT
    if args.action=='run':root.mkdir()
    ba.write(root/'inputs.json',campaign);results={};started=time.monotonic()
    try:
        for arm in CASES if campaign['readiness']=='ready' else ('parameters',):
            free=shutil.disk_usage(ba.BASE).free
            if free<LIMITS['min_free_bytes']:raise RuntimeError('At least 60 GiB free required before each arm')
            case=root/arm;case.mkdir()
            m={k:campaign[k] for k in ('parameters','limits','runtime','undistort_options','readiness')}
            m.update(arm=arm,worker_bindings=dict(campaign['worker_common_bindings']))
            if arm in CASES:
                source=campaign['sources'][arm];m['source']=source
                for k in ('names','images','feature_counts','candidate_pairs','input_manifest_sha256','source_inputs_sha256'):m[k]=campaign[k]
                m['worker_bindings'].update(campaign['image_bindings']);m['worker_bindings'].update(source['model_hashes'])
                other=campaign['sources'][CASES[1] if arm==CASES[0] else CASES[0]]
                m['denied_inputs']=[str(Path(other['model'])/'cameras.bin'),str(full.U/'Image_orientations_dataset1.xyz')]
                fixed.DATABASE=SOURCE/'matches/snapshot.db'
            ba.write(case/'inputs.json',m);probe=fixed.isolate(case,m)
            command=['/usr/bin/env','-i','PATH=/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE=1','TMPDIR='+str(case),
                'OPENSSL_CONF=/dev/null','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','VECLIB_MAXIMUM_THREADS=1',
                '/usr/bin/sandbox-exec','-f',probe['profile'],str(ba.RUNTIME/'bin/python'),'-B',str(ENTRY),
                '_check_worker' if args.action=='check' else '_worker',str(case/'inputs.json')]
            ba.write(case/'command.json',{'argv':command,'inputs_sha256':ba.sha(case/'inputs.json'),'profile_sha256':ba.sha(probe['profile']),'limits':LIMITS})
            status=ba.supervise(command,case,timeout=LIMITS['timeout_seconds'],max_rss=LIMITS['max_rss_bytes'])
            status['final_log_hashes']={str(p):ba.sha(p) for p in case.rglob('*') if p.is_file() and p.suffix in ('.log','.stdout','.stderr')}
            ba.write(case/'supervision.json',status);ba.verify(campaign['bindings'])
            if status['returncode'] or status['reason'] is not None:raise RuntimeError(f'{arm} failed; retain outputs: {status}')
            result=ba.read_json(case/'worker-result.json')
            result.update(supervision=status,isolation=probe,inputs_sha256=ba.sha(case/'inputs.json'),output_hashes={str(p):ba.sha(p) for p in case.rglob('*') if p.is_file()})
            ba.write(case/'result.json',result);results[arm]={'path':str(case/'result.json'),'sha256':ba.sha(case/'result.json'),'status':result['status']}
        final={'status':'completed' if args.action=='run' else 'checked','source_readiness':campaign['readiness'],'arms':results,'elapsed_seconds':time.monotonic()-started}
        ba.write(root/'result.json',final);print(json.dumps({'root':str(root),**final}))
    except BaseException as exc:
        ba.write(root/'failure.json',{'status':'failed','error':repr(exc),'completed_arms':results});raise


if __name__=='__main__':main()
