import importlib.util
import pytest
import torch
from torch import nn

def api(name):
    assert importlib.util.find_spec('reproduce.'+name) is not None,name+' is not implemented'
    return __import__('reproduce.'+name,fromlist=['*'])

def test_frozen_analyzer_keeps_input_derivative_and_bn_state():
    a=api('tasks'); m=a.freeze(nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Linear(4,2)))
    x=torch.randn(2,3,requires_grad=True);before=m[1].running_mean.clone()
    m(x).square().mean().backward()
    assert x.grad.abs().sum()>0 and all(p.grad is None for p in m.parameters())
    assert torch.equal(before,m[1].running_mean)

def test_likelihood_rate_has_correct_pixel_units():
    p=api('image_proxy')
    out={'likelihoods':{'y':torch.full((1,2,4,4),.5),'z':torch.full((1,1,2,2),.25)}}
    assert p.likelihood_bpp(out,8,8)==pytest.approx((32+8)/64)

def test_image_and_video_losses_follow_different_paper_equations():
    t=api('training'); x=torch.zeros(1,1,3,8,8);y=torch.ones_like(x)*.25
    rate=torch.tensor(2.);task=torch.tensor(3.)
    assert t.image_objective(rate,task,x,y,35)==pytest.approx(2+t.task_lambda(35)*3+.5*.25**2)
    assert t.video_objective(rate,task,x,y)==pytest.approx(10*(.25**2+.001*2)+3)

def test_phase_counters_do_not_turn_a_proxy_only_run_into_reproduction():
    t=api('training'); from reproduce.protocol import recipe
    cfg=recipe('od','full'); phases=t.phase_schedule(cfg)
    assert phases==[('proxy',400000),('fixed',120000),('adaptive',100000)]
    state=t.initial_state(cfg)
    state['phase_steps']['proxy']=400000
    assert t.scientific_complete(cfg,state) is False
    assert t.trained_npp_steps(state)==0

def test_four_qp_schedule_is_balanced_and_fixed_phase_stays_fixed():
    t=api('training'); from reproduce.protocol import recipe
    cfg=recipe('od','full')
    assert {t.qp_for_step(i,cfg,'proxy') for i in range(4)}=={30,35,40,45}
    assert {t.qp_for_step(i,cfg,'fixed') for i in range(4)}=={35}

def test_tracking_gaussian_loss_depends_on_the_ground_truth_location():
    t=api('tasks'); scores=torch.zeros(2,1,5,5,requires_grad=True)
    scores.data[:,:,2,2]=1
    centered=torch.tensor([[12.,12.,8.,8.],[12.,12.,8.,8.]])
    corner=torch.tensor([[0.,0.,4.,4.],[0.,0.,4.,4.]])
    a=t.tracking_score_loss(scores,centered,(32,32)); b=t.tracking_score_loss(scores,corner,(32,32))
    assert a<b
    a.backward();assert torch.isfinite(scores.grad).all()
