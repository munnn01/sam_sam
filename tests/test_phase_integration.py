import json
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
torch.set_num_threads(2)

def test_current_cuda_cpp_api_replaces_removed_thc_interface():
    from reproduce import assets
    assert hasattr(assets,'modernize_prroi_source'),'legacy native extension compatibility is not implemented'
    source='#include <THC/THC.h>\nTHCudaCheck(cudaGetLastError());\nauto p=x.data<float>();\nauto s=at::cuda::getCurrentCUDAStream();'
    result=assets.modernize_prroi_source(source)
    assert 'THC/THC.h' not in result and 'THCudaCheck' not in result
    assert 'C10_CUDA_CHECK' in result and '.data_ptr<float>()' in result
    assert 'c10::cuda::getCurrentCUDAStream' in result

def test_video_phase_optimizer_state_does_not_retain_prior_optimizer(tmp_path):
    from reproduce.protocol import recipe
    from reproduce.models import VideoNPP
    from reproduce.virtual_codec import VirtualCodec
    from reproduce.data import got_records
    from reproduce.tasks import freeze,fingerprint
    from reproduce.training import train,trained_npp_steps
    folder=tmp_path/'GOT-10k_Train_000001';folder.mkdir()
    for i in range(2):Image.fromarray(np.random.default_rng(i).integers(0,255,(16,16,3),dtype=np.uint8)).save(folder/f'{i+1:08}.jpg')
    (folder/'groundtruth.txt').write_text('3,3,8,8\n3,3,8,8\n')
    class SmallSupervisedAnalyzer:
        def __init__(self):
            self.model=freeze(nn.Linear(3,4));self.weight_digest=fingerprint(self.model)
        def receipt(self):return {'task':'ot','test_network':'linear supervised box regressor'}
        def loss(self,x,target):
            features=x.mean(dim=(1,3,4));truth=target['boxes_xywh'].mean(dim=0)[None]/16
            return (self.model(features)-truth).square().mean()
    cfg=recipe('ot','pilot');cfg.update(prior_steps=1,adaptive_steps=2,npp_size=16,frames=2,log_every=1,checkpoint_every=1)
    a=SmallSupervisedAnalyzer();npp=VideoNPP(width=4);proxy=VirtualCodec()
    # nn + data + optimizer + structured codec all execute; no task downloads or mocked calls.
    out=tmp_path/'run';out.mkdir()
    state,model=train(cfg,got_records(tmp_path,'train'),npp,proxy,a,None,out)
    checkpoint=torch.load(out/'last.pt',weights_only=False)
    assert trained_npp_steps(state)==2 and state['selected_step']==2
    assert set(checkpoint['optimizers'])=={'npp'}
    assert all(not p.requires_grad for p in proxy.parameters())
    assert fingerprint(a.model)==a.weight_digest

def test_zero_remaining_session_time_preserves_resume_without_exporting_untrained_npp(tmp_path):
    import inspect
    from reproduce.training import train,trained_npp_steps
    from reproduce.models import VideoNPP
    from reproduce.virtual_codec import VirtualCodec
    from reproduce.protocol import recipe
    assert 'max_seconds' in inspect.signature(train).parameters,'shared session deadline is not implemented'
    class Analyzer:
        model=nn.Linear(3,4)
        def receipt(self):return {'task':'ot'}
    cfg=recipe('ot','pilot');out=tmp_path/'budget';out.mkdir()
    state,npp=train(cfg,[{'unused':'deadline expires first'}],VideoNPP(width=4),VirtualCodec(),Analyzer(),None,out,max_seconds=0)
    assert trained_npp_steps(state)==0 and state['selected_step'] is None
    assert (out/'last.pt').is_file() and not (out/'trained_npp.pt').exists()

def test_image_resume_at_fixed_boundary_does_not_load_fixed_optimizer_into_adaptive(tmp_path):
    from reproduce.training import train,initial_state,save_checkpoint
    from reproduce.models import ImageNPP
    from reproduce.protocol import recipe
    cfg=recipe('ic','pilot');cfg.update(proxy_steps=0,fixed_steps=1,adaptive_steps=1)
    fixed=ImageNPP();proxy=nn.Linear(1,1);state=initial_state(cfg);state['phase_steps']['fixed']=1
    optimizer=torch.optim.Adam(fixed.parameters(),lr=cfg['lr'])
    checkpoint=tmp_path/'fixed_end.pt';save_checkpoint(checkpoint,cfg,state,fixed,proxy,{'npp':optimizer})
    class Analyzer:
        model=nn.Linear(1,1)
        def receipt(self):return {'task':'ic'}
    out=tmp_path/'resume';out.mkdir()
    state,adaptive=train(cfg,[],ImageNPP(),nn.Linear(1,1),Analyzer(),None,out,resume=checkpoint,max_seconds=0)
    assert adaptive.adaptive and state['phase_steps']=={'proxy':0,'fixed':1,'adaptive':0}

def test_long_video_preprocessing_can_offload_chunk_outputs_to_cpu():
    from reproduce.evaluation import processed_video
    class Network(nn.Module):
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(.5));self.seen=[]
        def forward(self,x,qp):self.seen.append(x.shape[1]);return x*self.weight
    model=Network();x=torch.rand(1,9,3,8,8)
    result=processed_video(model,x,35,4,output_device='cpu')
    assert model.seen==[4,4,1] and result.device.type=='cpu'
    assert torch.allclose(result,x*.5)
