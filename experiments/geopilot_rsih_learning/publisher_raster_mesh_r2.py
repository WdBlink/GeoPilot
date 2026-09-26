"""Bound thin repair of COLMAP 3.12.6 zero-distortion undistortion model skip."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
DRIVER = ENTRY.with_name('publisher_raster_mesh.py')
DRIVER_SHA = '973aab47b2ca1ace4d92a643f8f1d4e5c587ea8e4b0c1bda5429a710ae2bc37a'
if hashlib.sha256(DRIVER.read_bytes()).hexdigest() != DRIVER_SHA:
    raise ValueError('Frozen original mesh driver changed')
spec = importlib.util.spec_from_file_location('frozen_publisher_mesh', DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
pc, np, ba = driver.pc, driver.np, driver.ba
BASE = ba.BASE
PLAN = BASE/'publisher-raster-mesh-r2-plan.md'
ORIGINAL_SELF_TEST = driver.self_test
ORIGINAL_INPUTS = driver.inputs
ORIGINAL_UNDISTORT = pc.undistort_images


def image_size(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(24)
        if prefix[:8] == b'\x89PNG\r\n\x1a\n':
            return struct.unpack('>II', prefix[16:24])
        stream.seek(0)
        if stream.read(2) != b'\xff\xd8':
            raise ValueError('Expected JPEG/PNG header')
        for _ in range(4096):
            if stream.read(1) != b'\xff': raise ValueError('Invalid JPEG marker')
            code = stream.read(1)
            while code == b'\xff': code = stream.read(1)
            if not code or code[0] in (0xd9, 0xda): raise ValueError('Missing JPEG SOF')
            marker = code[0]
            if marker == 1 or 0xd0 <= marker <= 0xd7: continue
            size = int.from_bytes(stream.read(2), 'big')
            if size < 2: raise ValueError('Invalid JPEG segment')
            data = stream.read(size-2)
            if len(data) != size-2: raise ValueError('Truncated JPEG')
            if marker in (0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf):
                return int.from_bytes(data[3:5], 'big'), int.from_bytes(data[1:3], 'big')
        raise ValueError('JPEG header bound exceeded')


def camera_equal(a, b):
    return (a.model == b.model and (a.width,a.height) == (b.width,b.height)
            and np.array_equal(a.params, b.params))


def geometry(model):
    # Excludes the only allowed changes: camera intrinsics and image-plane xy.
    return {'cameras': sorted(model.cameras), 'rigs': {str(i): r.todict() for i,r in model.rigs.items()},
        'images': {str(i): {'name': im.name, 'camera': im.camera_id, 'frame': im.frame_id,
            'pose': im.cam_from_world().matrix().tolist(), 'point_ids': [p.point3D_id for p in im.points2D]}
            for i,im in model.images.items()},
        'frames': {str(i): f.todict() for i,f in model.frames.items()},
        'points': {str(i): {'xyz': p.xyz.tolist(), 'color': p.color.tolist(), 'error': p.error,
            'track': [(e.image_id,e.point2D_idx) for e in p.track.elements]} for i,p in model.points3D.items()}}


def normalized_geometry(model):
    return driver.fixed.plain(geometry(model))


def reconcile(source, output, options):
    folder = Path(output)/'sparse'
    model = pc.Reconstruction(str(folder))
    baseline = normalized_geometry(model)
    if baseline != normalized_geometry(source):
        raise ValueError('Native undistortion changed poses/3D/tracks/IDs')
    expected = {i: pc.undistort_camera(options,c) for i,c in source.cameras.items()}
    repairs, rows = [], []
    before = Path(output)/'native-model-before-repair'
    shutil.copytree(folder, before)
    native_hashes = {str(p): ba.sha(p) for p in before.iterdir() if p.is_file()}
    for ident, camera in source.cameras.items():
        native, target = model.cameras[ident], expected[ident]
        target.camera_id = ident
        affected = not camera_equal(native, target)
        if affected:
            zero = (camera.model.name in ('PINHOLE','SIMPLE_PINHOLE') or
                    (camera.model.name == 'SIMPLE_RADIAL' and camera.params[3] == 0.))
            if not zero or not camera_equal(native,camera):
                raise ValueError('Unexpected native camera discrepancy, not the known zero-distortion skip')
            repairs.append(ident)
        for image_id, original in source.images.items():
            if original.camera_id != ident: continue
            image = model.images[image_id]
            xy = np.asarray([p.xy for p in original.points2D]).reshape(-1,2)
            rays = camera.cam_from_img(xy)
            mapped = target.img_from_cam(np.column_stack((rays,np.ones(len(rays)))))
            if not np.isfinite(mapped).all(): raise ValueError('Nonfinite native image-plane map')
            native_xy = np.asarray([p.xy for p in image.points2D]).reshape(-1,2)
            roundtrip = camera.img_from_cam(np.column_stack((rays,np.ones(len(rays)))))
            skip_source_delta = float(np.max(np.abs(native_xy-xy))) if len(xy) else 0.
            skip_roundtrip_delta = float(np.max(np.abs(native_xy-roundtrip))) if len(xy) else 0.
            if affected:
                # Native still roundtrips image points even when its camera conversion is skipped.
                if not np.allclose(native_xy,roundtrip,rtol=0,atol=1e-9):
                    raise ValueError('Skipped native xy differs from source-camera native roundtrip')
                for point, value in zip(image.points2D, mapped): point.xy = value
            elif not np.allclose(native_xy,mapped,rtol=0,atol=1e-9):
                raise ValueError('Non-skipped native point map differs; refuse double transform')
            dimensions = image_size(Path(output)/'images'/image.name)
            if dimensions != (target.width,target.height) or max(dimensions) > options.max_image_size:
                raise ValueError('Actual raster header disagrees with expected native camera')
            rows.append({'image': image.name, 'camera_id': ident, 'header_dimensions':list(dimensions),
                         'repaired':affected, 'points2D':len(xy),
                         'max_skipped_native_source_delta_px':skip_source_delta if affected else None,
                         'max_skipped_native_roundtrip_delta_px':skip_roundtrip_delta if affected else None})
        if affected: model.cameras[ident] = target
    if normalized_geometry(model) != baseline: raise ValueError('Repair changed protected geometry')
    for ident, target in expected.items():
        if not camera_equal(model.cameras[ident],target) or model.cameras[ident].model.name != 'PINHOLE':
            raise ValueError('Final camera is not the expected PINHOLE')
    model.write_binary(str(folder))
    final = pc.Reconstruction(str(folder))
    if normalized_geometry(final) != baseline: raise ValueError('Repaired model roundtrip changed geometry')
    for ident, target in expected.items():
        if not camera_equal(final.cameras[ident],target): raise ValueError('Camera roundtrip drift')
    for image_id, image in model.images.items():
        if not np.array_equal(np.asarray([p.xy for p in image.points2D]),
                              np.asarray([p.xy for p in final.images[image_id].points2D])):
            raise ValueError('Point2D roundtrip drift')
    report = {'status':'PASS', 'repair':'native-ray-map-zero-distortion-skip', 'repaired_camera_ids':repairs,
        'images':rows, 'native_before_hashes':native_hashes,
        'final_model_hashes':{str(p):ba.sha(p) for p in folder.iterdir() if p.is_file()},
        'source_geometry_unchanged':True, 'options':options.todict()}
    ba.write(Path(output)/'camera-raster-reconciliation.json',report)
    return report


def undistort_images(output_path, input_path, image_path, **kwargs):
    source = pc.Reconstruction(str(input_path))
    source_hashes = {str(p):ba.sha(p) for p in Path(input_path).iterdir() if p.is_file()}
    ORIGINAL_UNDISTORT(output_path,input_path,image_path,**kwargs)
    report = reconcile(source,output_path,kwargs['undistort_options'])
    ba.verify(source_hashes)
    return report


def inputs():
    manifest = ORIGINAL_INPUTS()
    pins = {str(DRIVER):DRIVER_SHA, str(BASE/'publisher-raster-mesh-plan.md'):
            'f77076efe708d0ea0aa82bf64536a34228a0a0bb670708901d7a072ce459416a'}
    manifest['bindings'].update(pins)
    manifest['worker_common_bindings'].update(pins)
    manifest['repair'] = {'entry':str(ENTRY),'original_driver_sha256':DRIVER_SHA,
        'failed_run_preserved':str(BASE/'publisher-raster-mesh-run'),
        'policy':'native expected PINHOLE and ray mapping only on demonstrated skipped zero-distortion camera'}
    ba.verify(pins)
    return manifest


def png(path):
    def chunk(name,data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    raw=b''.join(
        b'\x00'+b''.join(bytes((x*3,y*4,120)) for x in range(80)) for y in range(60))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',80,60,8,2,0,0,0))+
                     chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))


def regression():
    root=Path(tempfile.mkdtemp(prefix='publisher-raster-mesh-r2-regression-')).resolve()
    reports={}
    for name,k in [('zero',0.),('distorted',.03)]:
        case=root/name; (case/'images').mkdir(parents=True); (case/'model').mkdir()
        model=pc.Reconstruction()
        world=np.array([[.103719,.1873,2.317],[1e-8,-1e-7,1.291],[-.739177,-.53913,1.573]])
        for ident in (1,2):
            camera=pc.Camera(camera_id=ident,model='SIMPLE_RADIAL',width=80,height=60,params=[69.713,39.137,29.791,k])
            model.add_camera(camera); rig=pc.Rig(rig_id=ident);rig.add_ref_sensor(camera.sensor_id);model.add_rig(rig)
            pose=pc.Rigid3d(np.column_stack((np.eye(3),[-(ident-1)*.1,0.,0.])))
            xy=camera.img_from_cam(world+np.array([-(ident-1)*.1,0.,0.]))
            im=pc.Image(name=f'{ident}.png',keypoints=xy,camera_id=ident,image_id=ident);im.frame_id=ident
            frame=pc.Frame(frame_id=ident,rig_id=ident,rig_from_world=pose);frame.add_data_id(im.data_id)
            model.add_frame(frame);model.add_image(im);model.register_frame(ident);png(case/'images'/im.name)
        for index,xyz in enumerate(world):
            track=pc.Track();track.add_element(1,index);track.add_element(2,index)
            model.add_point3D(xyz,track,np.array([120,100,80],dtype=np.uint8))
        model.write_binary(str(case/'model'))
        report=undistort_images(str(case/'undistorted'),str(case/'model'),str(case/'images'),
            output_type='COLMAP',copy_policy=pc.CopyType.copy,num_patch_match_src_images=20,
            undistort_options=pc.UndistortCameraOptions(max_image_size=40))
        if (name=='zero') != bool(report['repaired_camera_ids']):raise AssertionError('Wrong repair path')
        if name=='zero' and not any(r['max_skipped_native_source_delta_px'] > 0 for r in report['images']):
            raise AssertionError('Toy failed to exercise observed non-bitexact native roundtrip')
        result=pc.Reconstruction(str(case/'undistorted/sparse'))
        for im in result.images.values():
            for p in im.points2D:
                projected=result.cameras[im.camera_id].img_from_cam(im.cam_from_world()*result.points3D[p.point3D_id].xyz)
                if not np.allclose(projected,p.xy,atol=1e-8,rtol=0):raise AssertionError('Reprojection mismatch')
        command=[str(ba.legacy.TOOLS/'InterfaceCOLMAP'),'-i',str(case/'undistorted'),'-o',str(case/'scene.mvs'),
            '--image-folder',str(case/'undistorted/images'),'--archive-type','2','--max-threads','1']
        with (case/'interface.stdout').open('xb') as out,(case/'interface.stderr').open('xb') as err:
            process=subprocess.run(command,cwd=case,stdout=out,stderr=err,timeout=30)
        if process.returncode or not (case/'scene.mvs').is_file():raise ValueError('Tiny InterfaceCOLMAP failed')
        reports[name]={'reconciliation':report,'interface_argv':command,'interface_returncode':process.returncode}
    ba.write(root/'result.json',{'status':'PASS','synthetic_only':True,'reports':reports,
        'entry_sha256':ba.sha(ENTRY),'plan_sha256':ba.sha(PLAN),'driver_sha256':DRIVER_SHA,
        'hashes':{str(p):ba.sha(p) for p in root.rglob('*') if p.is_file()}})
    print(json.dumps({'status':'PASS','root':str(root)}))


def self_test():
    proof = ORIGINAL_SELF_TEST()
    if sys.argv[1:2] == ['_check_worker']:
        regression()
    return proof+'; r2 synthetic regression is run in check worker'


driver.self_test=self_test
driver.ENTRY=ENTRY
driver.PLAN=PLAN
driver.OUTPUT=BASE/'publisher-raster-mesh-r2-run'
driver.inputs=inputs
pc.undistort_images=undistort_images
if __name__=='__main__':
    if not __debug__ or Path(sys.prefix).resolve()!=ba.RUNTIME.resolve():raise RuntimeError('Use baseline-runtime -B')
    if sys.argv[1:]==['regression']: regression()
    else: driver.main()
