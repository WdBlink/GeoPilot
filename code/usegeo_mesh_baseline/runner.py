"""Sparse RGB SfM -> allowed-center alignment -> 2.5D integration mesh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
import numpy as np
import pycolmap as pc
import scipy
from scipy.spatial import Delaunay

CONFIG = {'method':'rgb-sparse-sfm-xy-mesh-v1','seed':917,'threads':1,
          'max_image_size':1600,'max_num_features':8192,'sequential_overlap':5,
          'spatial_neighbors':10,'spatial_max_distance_m':1e9,
          'timeout_seconds':14400,'max_rss_bytes':16*1024**3,
          'camera_model':'SIMPLE_RADIAL','camera_mode':'PER_IMAGE'}
HEADER = (b'ply\nformat binary_little_endian 1.0\nelement vertex {v}\nproperty double x\n'
          b'property double y\nproperty double z\nelement face {f}\n'
          b'property list uchar int vertex_indices\nend_header\n')


def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, default=str, allow_nan=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def unique_json(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key')
        obj[key]=value
    return obj


def regular(path,root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe input file')
    return path


def options():
    return {'reader':pc.ImageReaderOptions(camera_model='SIMPLE_RADIAL'),
            'extraction':pc.SiftExtractionOptions(num_threads=1,use_gpu=False,max_image_size=1600,max_num_features=8192),
            'matching':pc.SiftMatchingOptions(num_threads=1,use_gpu=False),
            'sequential':pc.SequentialMatchingOptions(overlap=5,quadratic_overlap=False,expand_rig_images=False,loop_detection=False,num_threads=1),
            'spatial':pc.SpatialMatchingOptions(ignore_z=False,max_num_neighbors=10,max_distance=1e9,num_threads=1),
            'verification':pc.TwoViewGeometryOptions(),
            'mapper':pc.IncrementalPipelineOptions(num_threads=1,ba_use_gpu=False,use_prior_position=False)}


def fit_centers(source,target):
    source,target=np.asarray(source,dtype=float),np.asarray(target,dtype=float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<3:
        raise ValueError('At least 3 corresponding centers required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite centers')
    if min(np.linalg.matrix_rank(source-source.mean(0)),np.linalg.matrix_rank(target-target.mean(0)))<2:
        raise ValueError('Degenerate center geometry')
    transform=pc.estimate_sim3d(source,target)
    if transform is None or not np.isfinite(transform.matrix()).all() or transform.scale<=0:
        raise ValueError('Invalid Sim3')
    if abs(np.linalg.det(transform.rotation.matrix())-1)>1e-8:
        raise ValueError('Improper rotation')
    return transform,np.linalg.norm(transform*source-target,axis=1)


def make_mesh(points,point_ids,track_lengths):
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)!=len(point_ids) or len(points)!=len(track_lengths):
        raise ValueError('Invalid sparse points')
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite sparse points')
    order=sorted(range(len(points)),key=lambda i:(-int(track_lengths[i]),int(point_ids[i])))
    keep={}
    for i in order:
        keep.setdefault(tuple(points[i,:2]),i)
    indices=sorted(keep.values(),key=lambda i:int(point_ids[i]))
    vertices=points[indices]
    if len(vertices)<3 or np.linalg.matrix_rank(vertices[:,:2]-vertices[:,:2].mean(0))<2:
        raise ValueError('Insufficient XY geometry')
    triangles=Delaunay(vertices[:,:2]).simplices.astype('<i4')
    areas=np.linalg.norm(np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]]),axis=1)/2
    if not np.isfinite(areas).all() or np.any(areas<=0):
        raise ValueError('Invalid Delaunay triangle')
    return vertices,triangles


def write_mesh(path,vertices,triangles):
    if len(vertices)>np.iinfo(np.int32).max:
        raise ValueError('Too many vertices')
    faces=np.empty(len(triangles),dtype=[('count','u1'),('indices','<i4',(3,))])
    faces['count']=3
    faces['indices']=triangles
    with Path(path).open('xb') as f:
        f.write(HEADER.replace(b'{v}',str(len(vertices)).encode()).replace(b'{f}',str(len(triangles)).encode()))
        f.write(np.asarray(vertices,dtype='<f8').tobytes())
        f.write(faces.tobytes())


def native_pairs(db,opts):
    sequential=pc.SequentialPairGenerator(opts['sequential'],db).all_pairs()
    spatial=pc.SpatialPairGenerator(opts['spatial'],db).all_pairs()
    normalize=lambda pairs: sorted({tuple(sorted((int(a),int(b)))) for a,b in pairs if a!=b})
    return {'sequential':normalize(sequential),'spatial':normalize(spatial),'union':normalize(sequential+spatial)}


def worker(inputs,out):
    manifest_path=inputs/'input_manifest.json'
    regular(manifest_path,inputs)
    manifest=json.loads(manifest_path.read_text(),object_pairs_hook=unique_json)
    if manifest['track']!='rgb-oriented' or manifest['scene_id'] not in ['Dataset-1','Dataset-2','Dataset-3']:
        raise ValueError('Wrong input track/scene')
    entries=manifest['images']
    names=sorted(entry['image_id'] for entry in entries)
    if len(names)!=len(set(names)) or len(names)!=manifest['image_count']:
        raise ValueError('Duplicate/missing image entries')
    expected={entry['image_id']:entry['sha256'] for entry in entries}
    image_hashes={}
    for name in names:
        p=inputs/'images'/name
        if Path(name).name!=name or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(inputs):
            raise ValueError('Unsafe image input')
        image_hashes[name]=sha(p)
        if image_hashes[name]!=expected[name]:
            raise ValueError('Image hash mismatch')
    scene=manifest['scene_id'].split('-')[1]
    orientation=inputs/f'Image_orientations_dataset{scene}.xyz'
    regular(orientation,inputs)
    if sha(orientation)!=manifest['orientation_sha256']:
        raise ValueError('Orientation hash mismatch')
    centers={}
    for line in orientation.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            fields=line.split()
            if len(fields)<4 or fields[0] in centers:
                raise ValueError('Invalid/duplicate orientation row')
            centers[fields[0]]=np.array([float(x) for x in fields[1:4]])
    if any(n not in centers or not np.isfinite(centers[n]).all() for n in names):
        raise ValueError('Missing/invalid allowed centers')
    dump(out/'inputs.json',{'input_manifest_sha256':sha(manifest_path),'orientation_sha256':sha(orientation),
                           'images':image_hashes,'scene':manifest['scene_id'],'track':'rgb-oriented'})
    opts=options()
    opts['mapper'].mapper.num_threads=1
    opts['mapper'].image_names=names
    dump(out/'options.json',{key:value.todict() for key,value in opts.items()})
    pc.set_random_seed(917)
    database=str(out/'database.db')
    pc.extract_features(database,str(inputs/'images'),image_names=names,camera_mode=pc.CameraMode.PER_IMAGE,
                        camera_model='SIMPLE_RADIAL',reader_options=opts['reader'],sift_options=opts['extraction'],device=pc.Device.cpu)
    db=pc.Database(database)
    try:
        images=db.read_all_images()
        for image in images:
            if image.name not in expected:
                raise ValueError('Unexpected database image')
            db.write_pose_prior(image.image_id,pc.PosePrior(centers[image.name],pc.PosePriorCoordinateSystem.CARTESIAN))
        dump(out/'cameras.json',[{'id':c.camera_id,'width':c.width,'height':c.height,'model':str(c.model),'params':c.params.tolist()} for c in db.read_all_cameras()])
        pairs=native_pairs(db,opts)
        dump(out/'pairs.json',{'image_names':{im.image_id:im.name for im in images},**pairs})
    finally:
        db.close()
    pc.match_sequential(database,sift_options=opts['matching'],matching_options=opts['sequential'],verification_options=opts['verification'],device=pc.Device.cpu)
    pc.match_spatial(database,sift_options=opts['matching'],matching_options=opts['spatial'],verification_options=opts['verification'],device=pc.Device.cpu)
    modeldir=out/'models'
    modeldir.mkdir()
    models=pc.incremental_mapping(database,str(inputs/'images'),str(modeldir),options=opts['mapper'])
    if not models:
        raise ValueError('No reconstructed model')
    model_id=min(models,key=lambda i:(-len(models[i].reg_image_ids()),i))
    model=models[model_id]
    registered=sorted(model.reg_image_ids(),key=lambda i:model.images[i].name)
    source=np.array([model.images[i].projection_center() for i in registered])
    target=np.array([centers[model.images[i].name] for i in registered])
    transform,residual=fit_centers(source,target)
    ids=sorted(model.points3D)
    vertices,triangles=make_mesh(transform*np.array([model.points3D[i].xyz for i in ids]),ids,[model.points3D[i].track.length() for i in ids])
    write_mesh(out/'mesh.ply',vertices,triangles)
    dump(out/'submission.json',{'schema_version':'usegeo-mesh-submission-1.0',
        'protocol_version':'usegeo-mesh-rgb-oriented-v1','track':'rgb-oriented','scene_id':manifest['scene_id'],
        'product':'mesh','geometry':'mesh.ply','configuration_id':CONFIG['method'],
        'method_metadata':{'family':'non-Agent','method_id':CONFIG['method'],'version':'1','seed':917,
                           'description':'own RGB sparse SfM 2.5D integration baseline; not publisher MVS'}})
    registered_names=[model.images[i].name for i in registered]
    dump(out/'method-result.json',{'method':CONFIG['method'],'scene':manifest['scene_id'],'model_id':model_id,
        'registered':registered_names,'unregistered':sorted(set(names)-set(registered_names)),
        'input_count':len(names),'registered_count':len(registered),'sparse_point_count':len(ids),
        'all_model_registered_counts':{i:len(m.reg_image_ids()) for i,m in models.items()},
        'vertices':len(vertices),'triangles':len(triangles),'mesh_sha256':sha(out/'mesh.ply'),
        'submission_sha256':sha(out/'submission.json'),
        'sim3_matrix':transform.matrix().tolist(),'sim3_scale':transform.scale,'center_residuals_m':residual.tolist(),
        'provenance':'own RGB sparse SfM; allowed-center fit; XY Delaunay; not publisher MVS fixture'})


def process_rss(pgid):
    listing=subprocess.run(['/bin/ps','-axo','pgid=,rss='],capture_output=True,text=True,check=True)
    return [int(line.split()[1])*1024 for line in listing.stdout.splitlines()
            if len(line.split())==2 and int(line.split()[0])==pgid]


def supervise(command,out,timeout=14400,max_rss=16*1024**3,rss_reader=process_rss):
    """Monitor one real worker process group; injectable measurement for fault checks."""
    start=time.monotonic()
    peak=0
    reason=None
    def cancelled(signum,frame):
        raise KeyboardInterrupt('Cancelled by signal')
    previous=signal.signal(signal.SIGTERM,cancelled)
    try:
        with (out/'worker.log').open('xb') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while proc.poll() is None:
                    elapsed=time.monotonic()-start
                    values=rss_reader(proc.pid)
                    if not values and proc.poll() is None:
                        raise RuntimeError('RSS unavailable')
                    peak=max(peak,sum(values))
                    if peak>max_rss:
                        raise RuntimeError('RSS_LIMIT')
                    if elapsed>timeout:
                        raise RuntimeError('TIME_LIMIT')
                    time.sleep(.5)
            except BaseException as exc:
                reason=f'{type(exc).__name__}: {exc}'
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait()
            returncode=proc.wait()
    except OSError as exc:
        returncode=-1
        reason=f'{type(exc).__name__}: {exc}'
    finally:
        signal.signal(signal.SIGTERM,previous)
    return {'returncode':returncode,'reason':reason,'wall_seconds':time.monotonic()-start,
            'peak_worker_process_group_rss_bytes':peak,'resource_sampling_seconds':.5}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--sandbox-profile',type=Path)
    args=parser.parse_args()
    if (platform.python_version(),pc.__version__,np.__version__,scipy.__version__)!=('3.10.20','3.12.6','1.26.4','1.11.1'):
        raise RuntimeError('Pinned runtime mismatch')
    inputs,out=args.inputs.resolve(),args.output.resolve()
    if args.worker:
        worker(inputs,out)
        return
    if args.sandbox_profile is None or not args.sandbox_profile.is_file():
        raise ValueError('An OS isolation profile is required for the method worker')
    if out.is_relative_to(inputs) or inputs.is_relative_to(out):
        raise ValueError('Inputs/output overlap')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'configuration.json',{'config':CONFIG,'source_sha256':sha(__file__),
                                  'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                  'cli_arguments':sys.argv,'cwd':os.getcwd(),
                                  'preregistration_sha256':sha(Path(__file__).with_name('PREREGISTRATION.md')),
                                  'sandbox_profile_sha256':sha(args.sandbox_profile),
                                  'versions':{'python':platform.python_version(),'pycolmap':pc.__version__,'numpy':np.__version__,'scipy':scipy.__version__}})
    # The observer stays outside the sandbox: macOS forbids setuid /bin/ps
    # inside it. Only this fixed, reference-isolated worker performs reconstruction.
    metrics=supervise(['/usr/bin/sandbox-exec','-f',str(args.sandbox_profile.resolve()),
                      sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--inputs',str(inputs),'--output',str(out)],out,
                      CONFIG['timeout_seconds'],CONFIG['max_rss_bytes'])
    success=metrics['returncode']==0 and metrics['reason'] is None and (out/'method-result.json').is_file()
    dump(out/'run.json',{'status':'SUCCESS' if success else 'FAILED',**metrics,'method':CONFIG['method']})
    if not success:
        raise SystemExit(1)


if __name__=='__main__':
    main()
