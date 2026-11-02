import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from reproduce.protocol import recipe

def test_full_recipe_evaluates_all_available_original_validation_sources():
    for task in ('od','ic','ar','ot'):
        assert recipe(task,'full')['validation_sources'] is None
        assert recipe(task,'pilot')['validation_sources']==8

def test_real_codec_evaluation_resumes_and_rejects_mixed_checkpoint(tmp_path):
    from reproduce.evaluation import evaluate
    from reproduce.codecs import RealCodec
    from reproduce.tasks import freeze,fingerprint
    ffmpeg=Path('D:/STUDY/AI/envs/ten_env/Library/bin/ffmpeg.exe')
    if not ffmpeg.is_file():pytest.skip('FFmpeg integration is checked in cloud preflight')
    class SmallNPP(nn.Module):
        def __init__(self):super().__init__();self.shift=nn.Parameter(torch.tensor(.05))
        def forward(self,x,qp):return (x+self.shift).clamp(0,1)
    class Classifier:
        def __init__(self):
            self.model=freeze(nn.Linear(3,6));self.weight_digest=fingerprint(self.model)
        def predict(self,x):return self.model(x.mean((1,3,4)))
    image=tmp_path/'sample.JPEG';Image.fromarray(np.random.default_rng(4).integers(0,255,(40,40,3),dtype=np.uint8)).save(image)
    rows=[dict(id='imagenet/heldout1',task='ic',path=str(image),label=0,split='val')]
    cfg=recipe('ic','pilot');cfg.update(npp_size=32,codecs=['h264'])
    npp=SmallNPP();a=Classifier();codec=RealCodec(ffmpeg=str(ffmpeg));out=tmp_path/'evaluation'
    import inspect
    assert 'max_seconds' in inspect.signature(evaluate).parameters,'evaluation continuation not implemented'
    partial=evaluate(cfg,rows,npp,a,codec,out,max_seconds=0)
    assert partial['complete'] is False
    full=evaluate(cfg,rows,npp,a,codec,out)
    assert full['complete'] is True and len(full['curves'])==8
    files=list(out.rglob('stream.h264'));times={str(p):p.stat().st_mtime_ns for p in files}
    cached=evaluate(cfg,rows,npp,a,codec,out,max_seconds=0)
    assert cached['complete'] is True
    assert times=={str(p):p.stat().st_mtime_ns for p in files}
    with torch.no_grad():npp.shift.add_(.1)
    with pytest.raises(ValueError):evaluate(cfg,rows,npp,a,codec,out)
