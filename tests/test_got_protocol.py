import numpy as np
from PIL import Image
import pytest
import torch

def test_got_metric_uses_visible_frames_after_initialization():
    from reproduce.evaluation import tracking_metrics
    gt=np.tile([1.,1.,4.,4.],(4,1));pred=gt.copy();pred[1]=[20,20,4,4];pred[3]=[20,20,4,4]
    import inspect
    assert 'cover' in inspect.signature(tracking_metrics).parameters,'official visibility mask is not implemented'
    scores=tracking_metrics(pred,gt,cover=[8,0,8,0],bound=(32,32))
    assert scores['frames']==1 and scores['ao']==pytest.approx(1.)

def test_got_aggregate_is_frame_weighted_like_official_toolkit():
    from reproduce import evaluation
    assert hasattr(evaluation,'aggregate_tracking'),'GOT frame-weighted aggregation is not implemented'
    result=evaluation.aggregate_tracking([dict(ao=1.,sr50=1.,sr75=1.,frames=1),dict(ao=0.,sr50=0.,sr75=0.,frames=3)])
    assert result['ao']==pytest.approx(.25)

def test_full_tracking_evaluation_loads_entire_annotated_sequence(tmp_path):
    from reproduce.data import got_records,load_sample
    seq=tmp_path/'GOT-10k_Val_000001';seq.mkdir()
    for i in range(5):Image.new('RGB',(40,20),(i*20,0,0)).save(seq/f'{i+1:08}.jpg')
    (seq/'groundtruth.txt').write_text('1,2,8,8\n'*5)
    (seq/'cover.label').write_text('8\n8\n0\n8\n8\n')
    rows=got_records(tmp_path,'val')
    x,target,meta=load_sample(rows[0],{'npp_size':32,'frames':0},False)
    assert x.shape[1]==5 and target['cover'].tolist()==[8,8,0,8,8]
