import importlib.util
import math
import pytest
import torch

def api(name):
    assert importlib.util.find_spec('reproduce') is not None, 'reproduction package is not implemented'
    return __import__('reproduce.'+name,fromlist=['*'])

def test_original_tasks_keep_user_qps():
    p=api('protocol')
    for task,paper in [('od','image'),('ic','image'),('ar','video'),('ot','video')]:
        cfg=p.recipe(task,'full')
        assert cfg['paper']==paper and cfg['qps']==[30,35,40,45]
        assert cfg['identity_fallback'] is False
        assert cfg['codecs']==(['bpg'] if paper=='image' else ['h264','h265'])

def test_proxy_lambda_matches_published_table_and_interpolates_log():
    p=api('protocol')
    assert p.proxy_lambda(34)==1024
    assert p.proxy_lambda(37)==512
    assert p.proxy_lambda(35)==pytest.approx(1024*2**(-1/3))
    assert p.proxy_lambda(45)==pytest.approx(128*2**(-1/3))

def test_real_forward_has_only_proxy_derivative():
    p=api('protocol')
    x=torch.tensor([2.],requires_grad=True)
    real=torch.tensor([7.],requires_grad=True)
    out=p.real_forward(3*x,real)
    assert out.item()==7
    out.sum().backward()
    assert x.grad.item()==3 and real.grad is None

def test_no_step_zero_or_nonfinite_checkpoint_selection():
    p=api('protocol')
    with pytest.raises(ValueError):p.validate_checkpoint(0,1.)
    with pytest.raises(ValueError):p.validate_checkpoint(1,float('nan'))
    assert p.validate_checkpoint(1,9.)==1  # Poor trained models are reported, never hidden by fallback.

def test_budget_stop_never_means_scientific_complete():
    p=api('protocol')
    r=p.run_receipt('ar',12,100,'budget_exhausted')
    assert r['scientific_complete'] is False and r['completed_steps']==12
    assert p.run_receipt('ar',100,100,'finished')['scientific_complete'] is True

def test_full_and_pilot_budgets_are_not_conflated():
    p=api('protocol')
    assert p.recipe('od','full')['proxy_steps']==400000
    assert p.recipe('od','full')['fixed_steps']==120000
    assert p.recipe('od','full')['adaptive_steps']==100000
    assert p.recipe('od','pilot')['scope']=='pilot_not_full_reproduction'
