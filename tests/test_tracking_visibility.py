import random
import numpy as np
from PIL import Image
import torch
from torch import nn

def test_training_crop_starts_visible_and_contains_a_visible_test_frame(tmp_path):
    from reproduce.data import got_records,load_sample
    seq=tmp_path/'GOT-10k_Train_000001';seq.mkdir()
    for i in range(5):Image.new('RGB',(32,32)).save(seq/f'{i+1:08}.jpg')
    (seq/'groundtruth.txt').write_text('4,4,8,8\n'*5)
    (seq/'cover.label').write_text('0\n0\n8\n8\n0\n')
    random.seed(2)
    x,target,meta=load_sample(got_records(tmp_path,'train')[0],{'npp_size':32,'frames':2},True)
    assert meta['start_frame']==2 and (target['cover']>0).all()

def test_tracking_task_loss_masks_absent_test_frames():
    from reproduce.tasks import Analyzer
    class Network(nn.Module):
        def forward(self,reference,tests,train_bb,proposals):
            self.test_count=len(tests)
            means=tests.mean((2,3,4))
            return [means[:,:,None,None].expand(-1,1,5,5)],means[:,:,None]
    analyzer=Analyzer.__new__(Analyzer);analyzer.task='ot';analyzer.device=torch.device('cpu')
    analyzer.model=Network();analyzer.mean=torch.zeros(1,3,1,1);analyzer.std=torch.ones(1,3,1,1)
    x=torch.rand(1,4,3,32,32,requires_grad=True)
    target={'boxes_xywh':torch.tensor([[4.,4.,8.,8.]]*4),'cover':torch.tensor([8.,0.,8.,0.])}
    loss=analyzer.loss(x,target);loss.backward()
    assert analyzer.model.test_count==1
    assert x.grad[:,2].abs().sum()>0 and x.grad[:,[1,3]].abs().sum()==0
