import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

def api(name):
    assert importlib.util.find_spec('reproduce.'+name) is not None, name+' is not implemented'
    return __import__('reproduce.'+name,fromlist=['*'])

def test_imagenet_never_invents_a_class_order_from_a_subset(tmp_path):
    d=api('data'); p=tmp_path/'n01440764';p.mkdir()
    Image.new('RGB',(35,48)).save(p/'one.JPEG')
    with pytest.raises(ValueError):d.imagenet_records(tmp_path,'train')
    rows=d.imagenet_records(tmp_path,'train',{'n01440764':0})
    assert rows[0]['label']==0 and rows[0]['split']=='train'

def test_coco_preserves_empty_targets_and_scales_boxes(tmp_path):
    d=api('data'); Image.new('RGB',(80,40)).save(tmp_path/'one.jpg')
    record={'task':'od','path':str(tmp_path/'one.jpg'),'id':'coco/1','image_id':1,
            'split':'train','width':80,'height':40,'objects':[[10,5,30,20,3]]}
    x,target,meta=d.load_sample(record,{'npp_size':32,'frames':1},training=False)
    assert x.shape==(1,1,3,32,64)
    assert torch.allclose(target['boxes'],torch.tensor([[8.,4.,24.,16.]]))
    record['objects']=[]
    assert d.load_sample(record,{'npp_size':32,'frames':1},False)[1]['boxes'].shape==(0,4)

def test_tracking_clip_uses_matching_gt_frames(tmp_path):
    d=api('data'); seq=tmp_path/'GOT-10k_Train_000001';seq.mkdir()
    for i in range(4):Image.new('RGB',(40,20),(i*30,0,0)).save(seq/f'{i+1:08}.jpg')
    (seq/'groundtruth.txt').write_text('1,2,10,5\n2,2,10,5\n3,2,10,5\n4,2,10,5\n')
    rows=d.got_records(tmp_path,'train')
    x,target,meta=d.load_sample(rows[0],{'npp_size':32,'frames':3},False)
    assert x.shape==(1,3,3,32,32)
    assert target['boxes_xywh'].shape==(3,4)
    assert torch.allclose(target['boxes_xywh'][:,0],torch.tensor([.8,1.6,2.4]))

def test_source_overlap_rejected_even_across_different_paths():
    d=api('data')
    with pytest.raises(ValueError):d.verify_splits([{'id':'same'}],[{'id':'same'}])

def test_got_index_is_lazy_but_loading_still_requires_aligned_frames(tmp_path):
    d=api('data');seq=tmp_path/'GOT-10k_Train_000001';seq.mkdir()
    (seq/'groundtruth.txt').write_text('1,2,10,5\n2,2,10,5\n')
    (tmp_path/'list.txt').write_text(seq.name+'\n')
    rows=d.got_records(tmp_path,'train')
    assert len(rows)==1
    with pytest.raises(ValueError):d.load_sample(rows[0],{'npp_size':32,'frames':2},False)

def test_real_h264_counts_full_stream_and_returns_exact_frames(tmp_path):
    c=api('codecs'); ffmpeg=Path('D:/STUDY/AI/envs/ten_env/Library/bin/ffmpeg.exe')
    if not ffmpeg.is_file():pytest.skip('FFmpeg integration runs on cloud preflight')
    codec=c.RealCodec(ffmpeg=str(ffmpeg),gop=2)
    x=torch.rand(1,2,3,32,48)
    out=codec(x,'h264',35,tmp_path)
    assert out.reconstruction.shape==x.shape
    assert out.bytes==Path(out.stream).stat().st_size
    assert out.bpp==pytest.approx(out.bytes*8/(2*32*48))
    assert out.sha256 and torch.isfinite(out.reconstruction).all()

def test_codec_rejects_video_in_image_bpg_mode():
    c=api('codecs')
    with pytest.raises(ValueError):c.RealCodec()(torch.rand(1,2,3,32,32),'bpg',35)
