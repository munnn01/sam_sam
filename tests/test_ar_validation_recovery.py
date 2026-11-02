import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from PIL import Image

from reproduce.protocol import recipe
from reproduce.training import config_digest


def test_completed_schedule_checkpoint_is_bound_to_source_config_and_bytes(tmp_path):
    assert importlib.util.find_spec('reproduce.ar_validation') is not None,'AR evaluation-only recovery is not implemented'
    from reproduce.ar_validation import verified_checkpoint
    cfg=recipe('ar','full')
    saved=dict(config=cfg,config_sha256=config_digest(cfg),source_manifest_sha256='a'*64,
               state=dict(phase_steps={'prior':512,'adaptive':100000},selected_step=100000,completed=False),npp={})
    checkpoint=tmp_path/'last.pt';torch.save(saved,checkpoint)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert verified_checkpoint(checkpoint,sha,'a'*64)['state']['phase_steps']['adaptive']==100000
    with pytest.raises(ValueError,match='SHA'):verified_checkpoint(checkpoint,'0'*64,'a'*64)
    with pytest.raises(ValueError,match='source'):verified_checkpoint(checkpoint,sha,'b'*64)
    saved['state']['phase_steps']['adaptive']=99999;torch.save(saved,checkpoint)
    with pytest.raises(ValueError,match='schedule'):
        verified_checkpoint(checkpoint,hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'a'*64)


def test_real_codec_short_clip_sampling_matches_both_arms_and_invalidates_old_cache(tmp_path):
    from reproduce.evaluation import evaluate
    from reproduce.codecs import RealCodec
    from reproduce.tasks import freeze,fingerprint
    ffmpeg=Path('D:/STUDY/AI/envs/ten_env/Library/bin/ffmpeg.exe')
    if not ffmpeg.is_file():pytest.skip('FFmpeg integration runs on Kaggle')
    class SmallNPP(nn.Module):
        def __init__(self):super().__init__();self.shift=nn.Parameter(torch.tensor(.02))
        def forward(self,x,qp):return (x+self.shift).clamp(0,1)
    class Classifier:
        def __init__(self):
            self.model=freeze(nn.Linear(3,6));self.weight_digest=fingerprint(self.model)
        def predict(self,x):return self.model(x.mean((1,3,4)))
    folder=tmp_path/'frames';folder.mkdir()
    for i in range(5):Image.new('RGB',(32,32),(i*30,40,80)).save(folder/f'img_{i+1:05}.jpg')
    rows=[dict(id='kinetics/short',task='ar',path=str(folder),label=0,split='val',source_format='provider_jpeg_frames')]
    cfg=recipe('ar','pilot');cfg.update(npp_size=32,codecs=['h264'],ar_validation_short_clip='loop')
    out=tmp_path/'evaluation';npp=SmallNPP();analyzer=Classifier();codec=RealCodec(ffmpeg=str(ffmpeg))
    result=evaluate(cfg,rows,npp,analyzer,codec,out)
    assert result['complete'] and len(result['curves'])==8
    expected=[(2*i)%5 for i in range(16)]
    for row in out.glob('cache/*/h264/*/00000000.json'):
        saved=json.loads(row.read_text())
        assert saved['sampling']['frame_indices']==expected
        assert saved['sampling']['short_clip_looped'] is True
    assert len(list(out.glob('cache/*/h264/*/00000000.json')))==8
    with pytest.raises(ValueError,match='different checkpoint'):
        evaluate(dict(cfg,ar_validation_short_clip='error'),rows,npp,analyzer,codec,out)


@pytest.mark.parametrize('old_identity',[{'npp_sha256':'other','config_sha256':'loop'},
                                         {'npp_sha256':'current','config_sha256':'old-sampling'},None])
def test_cache_restore_cannot_relabel_existing_checkpoint_or_protocol_results(tmp_path,old_identity):
    from reproduce import ar_validation
    assert hasattr(ar_validation,'restore_evaluation_cache'),'guarded cache restore is not implemented'
    source=tmp_path/'incoming';source.mkdir()
    identity={'npp_sha256':'current','config_sha256':'loop'}
    (source/'evaluation_identity.json').write_text(json.dumps(identity))
    destination=tmp_path/'existing';destination.mkdir()
    if old_identity is not None:(destination/'evaluation_identity.json').write_text(json.dumps(old_identity))
    stale=destination/'curve.json';stale.write_text('old results')
    before={p.name:p.read_bytes() for p in destination.iterdir()}
    with pytest.raises(ValueError,match='identity'):
        ar_validation.restore_evaluation_cache(source,destination)
    assert before=={p.name:p.read_bytes() for p in destination.iterdir()}


def test_cache_restore_allows_continuation_only_for_same_identity(tmp_path):
    from reproduce import ar_validation
    assert hasattr(ar_validation,'restore_evaluation_cache'),'guarded cache restore is not implemented'
    source=tmp_path/'incoming';source.mkdir()
    identity={'npp_sha256':'current','config_sha256':'loop'}
    (source/'evaluation_identity.json').write_text(json.dumps(identity))
    (source/'row.json').write_text('new row')
    destination=tmp_path/'existing';destination.mkdir()
    (destination/'evaluation_identity.json').write_text(json.dumps(identity))
    (destination/'curve.json').write_text('same-checkpoint result')
    ar_validation.restore_evaluation_cache(source,destination)
    assert (destination/'row.json').read_text()=='new row'
    assert (destination/'curve.json').read_text()=='same-checkpoint result'
