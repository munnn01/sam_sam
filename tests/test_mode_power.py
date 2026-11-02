import io,math
from types import SimpleNamespace
import pytest
import torch
from reproduce.codec_content_rate import ContentProxy,fit_mode_power_scales

class Bank(torch.nn.Module):
    def forward(self,x,q):return SimpleNamespace(reconstruction=x,bpp=x.mean()+.1)
class Codec:
    def overhead(self,x,q):return 10

def test_mode_fits_recover_distinct_positive_power_laws():
    rows=[]
    for mode,a,b in (('rounded',2.,1.5),('relaxed',.4,2.)):
        for q in (30,35,40,45):
            for raw in (.1,.2,.4,.8):
                rows.append(dict(qp=q,mode=mode,raw_bpp=raw,content_bpp=a*raw**b))
    fits=fit_mode_power_scales(rows)
    for mode,a,b in (('rounded',2.,1.5),('relaxed',.4,2.)):
        assert fits[mode]['scales']==pytest.approx([a]*4)
        assert fits[mode]['powers']==pytest.approx([b]*4)
    rows[0]['raw_bpp']=float('nan')
    with pytest.raises(ValueError):fit_mode_power_scales(rows)

def test_mode_dispatch_checkpoint_and_content_derivative():
    p=ContentProxy(Bank(),Codec())
    p.rounded_scale.fill_(2);p.rounded_power.fill_(1.5)
    p.relaxed_scale.fill_(.4);p.relaxed_power.fill_(2);p.separate_modes=True
    x=torch.full((1,1,3,4,8),.3,requires_grad=True)
    p.eval();a=p(x,35);assert float(a.content_bpp)==pytest.approx(2*.4**1.5)
    p.bank.train();b=p(x,35);assert float(b.content_bpp)==pytest.approx(.4*.4**2)
    b.bpp.backward();assert torch.allclose(x.grad,torch.full_like(x,.4*2*.4/x.numel()))
    data=io.BytesIO();torch.save(p.state_dict(),data);data.seek(0)
    clone=ContentProxy(Bank(),Codec());clone.load_state_dict(torch.load(data,weights_only=True))
    assert clone.separate_modes
    clone.eval();clone.bank.train()
    assert torch.equal(clone(x.detach(),35).bpp,b.bpp.detach())

def test_missing_modes_or_nonpositive_data_rejected():
    with pytest.raises(ValueError):fit_mode_power_scales([])
    rows=[dict(qp=q,mode=m,raw_bpp=.2,content_bpp=.3) for m in ('rounded','relaxed') for q in (30,35,40,45)]
    fit=fit_mode_power_scales(rows)
    assert fit['relaxed']['powers']==[1.]*4
    rows[-1]['content_bpp']=0
    with pytest.raises(ValueError):fit_mode_power_scales(rows)
