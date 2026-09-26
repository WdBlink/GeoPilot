"""Bind the unchanged U development scorer to the reconciled mesh-r2 producer."""
import hashlib
import importlib.util
from pathlib import Path

ENTRY = Path(__file__).resolve()
PARENT = ENTRY.with_name('publisher_raster_score.py')
PARENT_SHA = '4385502c1e55f24f0cdc595e5b75319ccb624f5776dc7ad566c1904bbe0aa9fa'
if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
    raise ValueError('Frozen U score parent changed')
spec = importlib.util.spec_from_file_location('frozen_u_score', PARENT)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
original_ready = score.ready
score.ENTRY = ENTRY
score.PLAN = score.BASE/'publisher-raster-score-r2-plan.md'
score.SOURCE = score.BASE/'publisher-raster-mesh-r2-run'
score.OUTPUT = score.BASE/'publisher-raster-r2-scores'
score.IDENTITY['producer_revision'] = 'mesh-r2'
score.PINS.update({
    PARENT: PARENT_SHA,
    ENTRY.with_name('publisher_raster_mesh_r2.py'):
        'ad504039c6004782aacb1f5edd5e41a03683955a55c1af6c9216fa735d173d46',
    score.BASE/'publisher-raster-mesh-r2-plan.md':
        '23971a59f1b883a8d52d4d046fa4bb9680955b08f0d74ce0f6135c961bce52e7',
})


def ready():
    source = original_ready()
    if source is None:
        return None
    for arm in score.ARMS:
        path = score.SOURCE/arm/'prepare/undistorted/camera-raster-reconciliation.json'
        if source['bindings'].get(str(path)) != score.sha(path):
            raise ValueError('Unbound native camera/raster reconciliation')
        proof = score.read(path)
        if proof['status'] != 'PASS' or proof['source_geometry_unchanged'] is not True:
            raise ValueError('Native camera/raster reconciliation failed')
        for model_file, digest in proof['final_model_hashes'].items():
            if source['bindings'].get(model_file) != digest:
                raise ValueError('Reconciled native model is not bound to mesh run')
    return source


score.ready = ready
if __name__ == '__main__':
    score.main()
