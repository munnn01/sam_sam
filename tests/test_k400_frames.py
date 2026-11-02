import csv
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from reproduce import data

def test_provider_validation_frames_join_official_labels_and_load_multiframe_clip(tmp_path):
    assert hasattr(data,'kinetics_frame_records'),'official-label frame validation adapter is not implemented'
    folder=tmp_path/'--7VUM9MKg4';folder.mkdir()
    for i in range(4):Image.new('RGB',(40,32),(i*25,0,0)).save(folder/f'img_{i+1:05}.jpg')
    annotation=tmp_path/'val.csv'
    annotation.write_text('label,youtube_id,time_start,time_end,split\nabseiling,--7VUM9MKg4,0,10,val\n')
    rows=data.kinetics_frame_records(tmp_path,annotation)
    assert len(rows)==1 and rows[0]['split']=='val' and rows[0]['id']=='kinetics/--7VUM9MKg4'
    x,label,meta=data.load_sample(rows[0],{'npp_size':32,'frames':2},False)
    assert x.shape==(1,2,3,32,32) and int(label)==0
    assert meta['source_format']=='provider_jpeg_frames' and meta['provider_cadence']=='unverified'

def test_official_got_sequence_list_does_not_stat_every_frame_directory(tmp_path):
    # Dataset mounts can expose a large manifest before per-file cold metadata is warm.
    # The loader, not the index, validates the selected source before use.
    (tmp_path/'list.txt').write_text('GOT-10k_Train_000001\nGOT-10k_Train_000002\n')
    rows=data.got_records(tmp_path,'train')
    assert len(rows)==2
    with pytest.raises((ValueError,OSError)):data.load_sample(rows[0],{'npp_size':32,'frames':2},False)


@pytest.mark.parametrize('available',[1,5,30,31,35])
def test_explicit_validation_loop_policy_keeps_short_sources_and_long_clip_sampling(tmp_path,available):
    folder=tmp_path/'frames';folder.mkdir()
    for i in range(available):
        Image.new('RGB',(32,32),(i*7,0,0)).save(folder/f'img_{i+1:05}.jpg')
    row=dict(id='kinetics/short',task='ar',path=str(folder),label=4,split='val',source_format='provider_jpeg_frames')
    cfg=dict(npp_size=32,frames=16,ar_validation_short_clip='loop')
    x,label,meta=data.load_sample(row,cfg,False)
    start=max(0,(available-31)//2)
    expected=[(start+2*i)%available for i in range(16)]
    assert x.shape==(1,16,3,32,32) and int(label)==4
    assert meta['frame_indices']==expected and meta['source_frames']==available
    assert meta['short_clip_looped']==(available<31)
    for i,index in enumerate(expected):
        original=data._tensor([data._rgb(folder/f'img_{index+1:05}.jpg')])
        assert np.allclose(x[:,i:i+1].numpy(),original.numpy())


def test_validation_loop_does_not_change_training_or_accept_empty_sources(tmp_path):
    row=dict(id='kinetics/empty',task='ar',path=str(tmp_path),label=0,split='val',source_format='provider_jpeg_frames')
    cfg=dict(npp_size=32,frames=16,ar_validation_short_clip='loop')
    with pytest.raises(ValueError,match='empty'):data.load_sample(row,cfg,False)
    Image.new('RGB',(32,32)).save(tmp_path/'img_00001.jpg')
    with pytest.raises(ValueError,match='too short'):data.load_sample(row,cfg,True)
