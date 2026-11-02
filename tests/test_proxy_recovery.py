import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from reproduce.models import VideoNPP
from reproduce.protocol import recipe
from reproduce.virtual_codec import VirtualCodec

def api():
    assert importlib.util.find_spec('reproduce.proxy_recovery') is not None,'recovery runner not implemented'
    return __import__('reproduce.proxy_recovery',fromlist=['*'])

def test_checkpoint_binding_requires_hash_source_and_completed_training(tmp_path):
    a=api();cfg=recipe('ot');source='a'*64
    saved=dict(config=cfg,source_manifest_sha256=source,
        state={'phase_steps':{'prior':512,'adaptive':100000}},
        npp=VideoNPP().state_dict(),proxy=VirtualCodec().state_dict())
    p=tmp_path/'last.pt';torch.save(saved,p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
    loaded=a.load_checkpoint(p,sha,source,'ot')
    assert loaded['state']['phase_steps']['adaptive']==100000
    with pytest.raises(ValueError):a.load_checkpoint(p,'b'*64,source,'ot')
    with pytest.raises(ValueError):a.load_checkpoint(p,sha,'b'*64,'ot')
    with pytest.raises(ValueError):a.load_checkpoint(p,sha,source,'ar')

def test_symbol_sampling_is_bounded_repeatable_and_preserves_channels():
    a=api();symbols=(torch.arange(192*8*8.).reshape(1,192,8,8),)*3
    x=a.sample_symbols(symbols,positions=17,seed=4)
    assert x.shape==(1,192,1,17)
    assert torch.equal(x,a.sample_symbols(symbols,17,4))

def test_pair_summary_does_not_treat_reference_as_actual_zero_bit_cost():
    a=api();from reproduce.proxy_variants import DiagnosticCodec
    output=DiagnosticCodec()(torch.full((1,2,3,8,8),.5),45)
    row=a.output_summary(output)
    assert row['excess_bpp']==pytest.approx(0,abs=2e-6)
    assert row['zero_fraction']==1
    assert row['zero_symbol_cost_bpp']==pytest.approx(row['entropy_bpp'])

def test_phase_budget_reports_incomplete_instead_of_training_after_deadline():
    a=api();budget=a.Budget(0)
    with pytest.raises(a.BudgetExpired):budget.require('paired')

def test_recovery_decision_blocks_lambda_when_any_paired_group_fails():
    a=api()
    assert a.recovery_decision({'passed':False})=='supervised_rate_head_required'
    assert a.recovery_decision({'passed':True})=='eligible_for_bounded_recovery'

def test_diagnostic_cli_defaults_to_no_npp_update():
    a=api();args=a.parser().parse_args(['--task','ot','--checkpoint','last.pt',
        '--checkpoint-sha256','a'*64,'--training-source-sha256','b'*64,'--out','out'])
    assert args.stage=='diagnose' and args.max_seconds==7200

def test_paired_tracker_rng_is_repeatable_and_restores_training_rng():
    a=api();import random
    random.seed(44);torch.manual_seed(44)
    python_before=random.getstate();torch_before=torch.get_rng_state().clone()
    with a.paired_rng('source'):
        first=(random.random(),torch.rand(3))
    assert random.getstate()==python_before and torch.equal(torch.get_rng_state(),torch_before)
    with a.paired_rng('source'):
        second=(random.random(),torch.rand(3))
    assert first[0]==second[0] and torch.equal(first[1],second[1])

def test_failed_checkpoint_cannot_inherit_stale_recovery_updates(tmp_path,monkeypatch):
    a=api();import sys
    (tmp_path/'lambda_runs.json').write_text(json.dumps([{'additional_steps':999}]))
    checkpoint=tmp_path/'last.pt';checkpoint.write_bytes(b'wrong')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(sys,'argv',['runner','--task','ot','--checkpoint',str(checkpoint),
        '--checkpoint-sha256','a'*64,'--training-source-sha256','b'*64,'--stage','recover','--out',str(tmp_path)])
    with pytest.raises(ValueError):a.main()
    report=json.loads((tmp_path/'run.json').read_text())
    assert report['npp_updates']==0 and 'recovery' not in report

def test_nested_rd_cohorts_reuse_individual_source_cache(tmp_path,monkeypatch):
    a=api();from types import SimpleNamespace
    cfg=recipe('ar');model=VideoNPP();rows=[dict(id=str(i),split='train',task='ar') for i in range(2)]
    monkeypatch.setattr(a,'load_probe',lambda row,cfg,frames,seed:(torch.zeros(1,frames,3,8,8),torch.tensor([0]),{}))
    class Codec:
        calls=0
        def __call__(self,x,cod,qp):
            self.calls+=1
            return SimpleNamespace(reconstruction=x,bytes=100+50-qp)
    codec=Codec();analyzer=SimpleNamespace(weight_digest='fixed',predict=lambda x:torch.tensor([[1.,0.]]))
    a.evaluate_probe(model,rows[:1],cfg,analyzer,codec,tmp_path,a.Budget(20),'one',arms=('codec_only',))
    a.evaluate_probe(model,rows,cfg,analyzer,codec,tmp_path,a.Budget(20),'two',arms=('codec_only',))
    assert codec.calls==16

def test_prior_comparison_accepts_one_shared_deadline_before_cache_work(tmp_path):
    a=api();from types import SimpleNamespace
    with pytest.raises(a.BudgetExpired):
        a.fit_candidates(VirtualCodec(),[],[],tmp_path,a.Budget(10),
            SimpleNamespace(lr_probe_steps=4,prior_updates=4),phase=a.Budget(0))

def test_real_gradient_audit_reports_eight_sample_means_for_output_and_parameters(tmp_path,monkeypatch):
    a=api();from types import SimpleNamespace
    from reproduce.proxy_variants import DiagnosticCodec
    cfg=recipe('ot');model=VideoNPP();proxy=DiagnosticCodec()
    monkeypatch.setattr(a,'load_probe',lambda *args:(torch.full((1,2,3,8,8),.51),None,{}))
    analyzer=SimpleNamespace(loss=lambda x,target:(x-.5).square().mean())
    report=a.gradient_audit(model,proxy,analyzer,[dict(id='probe')],cfg,tmp_path,a.Budget(30))
    noise=[row for row in report if row['mode']=='uniform_expected']
    assert len(noise)==2
    for row in noise:
        assert row['parameters']['noise_samples']==8 and row['output']['noise_samples']==8
        assert row['parameters']['noise_variance']>=0

def test_prior_candidate_screen_merges_real_fit_report_without_duplicate_lr(tmp_path):
    a=api();from types import SimpleNamespace
    def batches(source):return [dict(source=source,qp=q,kind='raw',codes=torch.zeros(1,192,1,8)) for q in (30,35,40,45)]
    result=a.fit_candidates(VirtualCodec(),batches('train'),batches('dev'),tmp_path,a.Budget(20),
        SimpleNamespace(lr_probe_steps=4,prior_updates=4),phase=a.Budget(20))
    assert [name for name,_,_ in result]==['shared','per_qp']
    assert len(json.loads((tmp_path/'prior_lr_screen.json').read_text()))==3
