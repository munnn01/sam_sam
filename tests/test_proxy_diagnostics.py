import importlib.util
import json
import pytest
import torch
from torch import nn
from reproduce.proxy_variants import DiagnosticCodec

def api():
    assert importlib.util.find_spec('reproduce.proxy_diagnostics') is not None,'diagnostic controls not implemented'
    return __import__('reproduce.proxy_diagnostics',fromlist=['*'])

def test_paired_gate_accepts_correct_direction_despite_large_static_offset():
    a=api();rows=[]
    for codec in ('h264','h265'):
        for qp in (30,35,40,45):
            for i in range(8):
                rows.append(dict(source=str(i),codec=codec,qp=qp,raw_bpp=.02,npp_bpp=.03,
                    proxy_raw=1.20,proxy_npp=1.21,pixels=100000))
    result=a.paired_gate(rows)
    assert result['passed'] and len(result['groups'])==8
    for row in rows:row['proxy_npp']=1.19
    assert a.paired_gate(rows)['passed'] is False

def test_gate_blocks_missing_qp_and_insufficient_pairs():
    a=api();row=dict(source='a',codec='h264',qp=45,raw_bpp=.02,npp_bpp=.03,proxy_raw=1.2,proxy_npp=1.21,pixels=1000)
    assert a.paired_gate([row])['passed'] is False

def test_rate_ao_margin_can_favor_a_lower_ao_at_lower_rate_and_never_extrapolates():
    a=api();baseline=[(.01,.5),(.02,.8),(.04,.9),(.08,.95)]
    result=a.rate_ao_margin(baseline,.015,.75)
    assert result['valid'] and result['margin_pp']>0
    assert a.rate_ao_margin(baseline,.001,.3)['valid'] is False

def test_cache_identity_rejects_relabeling_existing_results(tmp_path):
    a=api();p=tmp_path/'cache';a.bind_cache(p,{'npp':'one','sources':['a']})
    (p/'result.json').write_text('{}')
    with pytest.raises(ValueError):a.bind_cache(p,{'npp':'two','sources':['a']})

def test_calibration_split_is_source_disjoint_and_rejects_validation():
    a=api();rows=[dict(id=str(i),split='train') for i in range(20)]
    train,dev=a.calibration_split(rows,8,8,42)
    assert not ({r['id'] for r in train}&{r['id'] for r in dev})
    assert a.calibration_split(rows,8,8,42)==(train,dev)
    with pytest.raises(ValueError):a.calibration_split([dict(id='bad',split='val')]*20,8,8,42)

def test_prior_fit_is_bounded_improves_heldout_nll_and_rejects_overlap():
    a=api();model=DiagnosticCodec()
    def batches(source):return [dict(source=source,qp=q,kind='raw',codes=torch.zeros(1,192,1,8)) for q in (30,35,40,45)]
    report=a.fit_prior(model,batches('train'),batches('dev'),steps=8,lr=.01,eval_every=4,max_seconds=20)
    assert report['updates']==8 and report['best_loss']<report['initial_loss']
    with pytest.raises(ValueError):a.fit_prior(model,batches('same'),batches('same'),steps=2)
    stopped=a.fit_prior(model,batches('train'),batches('dev'),steps=100,max_seconds=0)
    assert stopped['updates']==0 and stopped['stop_reason']=='time_budget'

def test_per_qp_fit_accounts_for_total_updates_not_four_times_the_budget():
    a=api();model=DiagnosticCodec(per_qp=True)
    def batches(source):return [dict(source=source,qp=q,kind='raw',codes=torch.zeros(1,192,1,8)) for q in (30,35,40,45)]
    report=a.fit_prior(model,batches('train'),batches('dev'),steps=8,eval_every=4,max_seconds=20)
    assert sum(report['updates_by_qp'].values())==8
    assert set(report['updates_by_qp'].values())=={2}

def test_additional_steps_run_beyond_completed_schedule_with_fixed_lr_and_drift():
    a=api();model=nn.Linear(1,1,bias=False)
    def losses(npp,qp,step):
        value=npp(torch.ones(1,1)).square().mean()
        return dict(rate=value,distortion=value*.1,task=value)
    rows=[]
    report=a.finetune(model,losses,steps=3,lr=1e-4,rate_lambda=.01,original_steps=100000,log=rows.append)
    assert report['additional_steps']==3 and report['global_steps']==100003
    assert report['theta_delta_norm']>0
    assert len(rows)==3 and {r['lr'] for r in rows}=={1e-4}
    assert [r['qp'] for r in rows]==[30,35,40]

def test_component_gradient_report_measures_conflict_and_allows_zero_rate():
    a=api();p=nn.Parameter(torch.tensor(2.))
    report=a.component_gradients({'rate':p*0,'distortion':p.square(),'task':-p},[p])
    assert report['norms']['rate']==0 and report['norms']['task']==1
    assert report['cosines']['distortion/task']==pytest.approx(-1.)

def test_bootstrap_rd_margin_is_source_paired_and_no_ao_only_stop():
    a=api()
    def points(rate,quality):return [dict(source=str(i),bytes=rate*10000/8,pixels=10000,
        quality_sum=quality*20,quality_weight=20) for i in range(16)]
    base=[dict(points=points(r,q)) for r,q in [(.01,.5),(.02,.8),(.04,.9),(.08,.95)]]
    good=a.margin_interval(base,{'points':points(.015,.75)},samples=32)
    assert good['valid'] and good['low_pp']>0
    assert a.rd_should_stop([good,good],initial_margin_pp=-10) is False
    bad=dict(valid=True,margin_pp=-3.,low_pp=-4.,high_pp=-2.)
    assert a.rd_should_stop([bad,bad],initial_margin_pp=-3.)
    assert a.rd_should_stop([bad,bad],initial_margin_pp=-10.) is False
    altered=points(.015,.75);altered[0]['source']='different'
    with pytest.raises(ValueError):a.margin_interval(base,{'points':altered},samples=8)

def test_noise_gradient_summary_averages_vectors_before_norms():
    a=api();fixed={'task':torch.tensor([1.,0.]),'distortion':torch.tensor([0.,1.])}
    report=a.noise_gradient_summary([torch.tensor([1.,0.]),torch.tensor([-1.,0.])],fixed)
    assert report['norms']['rate']==0
    assert report['noise_sample_norm_mean']==1 and report['noise_variance']>0

def test_full_fft_residual_energy_is_orientation_independent():
    a=api();h,w=16,32
    low=torch.sin(2*torch.pi*torch.arange(w)/w)[None,:].expand(h,w)
    high=torch.sin(2*torch.pi*6*torch.arange(h)/h)[:,None].expand(h,w)
    x=torch.zeros(1,2,3,h,w);y=(low+high).expand_as(x)
    report=a.residual_statistics(x,y)
    assert report['high_frequency_fraction']==pytest.approx(.5,abs=2e-6)

def test_e0_calibration_requires_repeated_seeds_and_records_observed_spread():
    a=api();rows=[{'h264/40':{'valid':True,'margin_pp':v}} for v in (-1.,-2.,-.5)]
    assert a.calibrate_rd_tolerance(rows[:1])['available'] is False
    result=a.calibrate_rd_tolerance(rows)
    assert result['available'] and result['delta_pp']==pytest.approx(1.5)

def test_supplementary_progress_counts_survive_a_later_loss_failure():
    a=api();model=nn.Linear(1,1);seen=[]
    def loss(model,qp,index):
        if index==2:raise RuntimeError('probe failure')
        value=model(torch.ones(1,1)).square().mean()
        return dict(rate=value,distortion=value,task=value)
    with pytest.raises(RuntimeError):
        a.finetune(model,loss,steps=5,progress=lambda row,optimizer:seen.append(row.copy()))
    assert seen[-1]['additional_steps']==2 and seen[-1]['global_steps']==100002
