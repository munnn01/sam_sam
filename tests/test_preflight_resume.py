import pytest
import torch
from torch import nn

def test_preflight_rejects_missing_task_gradient_even_when_auxiliary_loss_is_differentiable():
    from reproduce import run
    assert hasattr(run,'gradient_receipt'),'isolated task gradient validation is missing'
    model=nn.Linear(2,1);y=model(torch.ones(1,2));task=y.detach().square().mean();total=y.square().mean()+task
    with pytest.raises(RuntimeError,match='task'):
        run.gradient_receipt(task,total,model)
    y=model(torch.ones(1,2));task=y.square().mean()
    result=run.gradient_receipt(task,task+y.mean(),model)
    assert result['task_npp_grad_norm']>0 and result['npp_grad_norm']>0

def test_resuming_validation_in_the_same_output_directory_preserves_cache(tmp_path):
    from reproduce import run
    assert hasattr(run,'restore_validation_cache'),'validation cache resume helper is missing'
    validation=tmp_path/'validation';validation.mkdir();saved=validation/'evaluation_identity.json';saved.write_text('{}')
    run.restore_validation_cache(tmp_path/'last.pt',tmp_path)
    assert saved.read_text()=='{}'
    destination=tmp_path/'other';destination.mkdir()
    run.restore_validation_cache(tmp_path/'last.pt',destination)
    assert (destination/'validation/evaluation_identity.json').read_text()=='{}'
