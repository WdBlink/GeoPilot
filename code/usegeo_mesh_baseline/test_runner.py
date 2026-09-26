"""Synthetic-only baseline checks; no real images or reference files."""
import tempfile
from pathlib import Path
import numpy as np
import pycolmap as pc
from runner import fit_centers, make_mesh, write_mesh, native_pairs, options, HEADER


def main():
    source=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[1,1,1.]])
    target=2*source+[498000,4379500,100]
    transform,residual=fit_centers(source,target)
    assert residual.max()<1e-8 and np.max(np.abs(transform*source-target))<1e-8
    try:
        fit_centers(np.zeros((3,3)),np.zeros((3,3)))
    except ValueError:
        pass
    else:
        raise AssertionError('Degenerate centers accepted')
    points=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,2]])
    vertices,triangles=make_mesh(points,[1,2,3,4,5],[2,2,2,2,3])
    assert len(vertices)==4 and len(triangles)==2 and any(np.array_equal(v,[0,0,2]) for v in vertices)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'mesh.ply'
        write_mesh(path,vertices,triangles)
        data=path.read_bytes()
        header=HEADER.replace(b'{v}',b'4').replace(b'{f}',b'2')
        assert data.startswith(header) and len(data)==len(header)+4*24+2*13
        decoded=np.frombuffer(data[len(header):len(header)+96],dtype='<f8').reshape(-1,3)
        assert np.array_equal(decoded,vertices)
        db=pc.Database(str(Path(tmp)/'synthetic.db'))
        try:
            camera=pc.Camera(model='SIMPLE_RADIAL',width=100,height=80,params=[120,50,40,0])
            camera_id=db.write_camera(camera)
            positions={}
            for i in range(14):
                image_id=db.write_image(pc.Image(name=f'{i:03d}.jpg',camera_id=camera_id))
                center=np.array([498000+i*i,4379500+i*3,100+i])
                positions[image_id]=center
                db.write_pose_prior(image_id,pc.PosePrior(center,pc.PosePriorCoordinateSystem.CARTESIAN))
            pairs=native_pairs(db,options())
            assert len(pairs['sequential'])==sum(14-d for d in range(1,6))
            assert all(a!=b for a,b in pairs['union'])
            expected=set()
            for i,c in positions.items():
                neighbors=sorted((j for j in positions if j!=i),key=lambda j:np.linalg.norm(positions[j]-c))[:10]
                expected.update(tuple(sorted((i,j))) for j in neighbors)
            assert set(pairs['spatial'])==expected
            assert set(pairs['union'])==set(pairs['sequential'])|set(pairs['spatial'])
        finally:
            db.close()
    print('PASS synthetic Sim3 / rank rejection / XY duplicate / strict PLY / native sequential-spatial pairs')


if __name__=='__main__':
    main()
