import math
from types import SimpleNamespace
import pytest
import torch
from reproduce.codec_content_rate import nal_accounting,ContentProxy,fit_scales,gradient_balance,screen_gate

def test_nal_bytes_accounted_exactly_for_both_codecs():
    avc=b'\x00\x00\x00\x01\x67abcd'+b'\x00\x00\x01\x68ef'+b'\x00\x00\x01\x65payload'
    r=nal_accounting(avc,'h264')
    assert r['content_bytes']==11 and r['overhead_bytes']==15 and r['total_bytes']==26
    hevc=b'\x00\x00\x01'+bytes([32<<1,1])+b'ab'+b'\x00\x00\x00\x01'+bytes([19<<1,1])+b'xyz'
    r=nal_accounting(hevc,'h265')
    assert r['content_bytes']==9 and r['overhead_bytes']==7 and r['total_bytes']==16
    with pytest.raises(ValueError):nal_accounting(b'not-annexb','h264')
    with pytest.raises(ValueError):nal_accounting(b'\x00\x00\x01\x67abcd','h264')

class Bank(torch.nn.Module):
    def forward(self,x,q):return SimpleNamespace(reconstruction=x,bpp=x.mean()+.1)
class Codec:
    def overhead(self,x,q):return 100
def test_overhead_scales_by_paid_pixels_without_image_gradient():
    p=ContentProxy(Bank(),Codec());p.content_scale.fill_(2)
    x=torch.full((1,1,3,10,20),.2,requires_grad=True)
    r=p(x,35)
    assert float(r.overhead_bpp)==4
    assert torch.allclose(r.bpp,2*(x.mean()+.1)+4)
    r.bpp.backward()
    assert torch.allclose(x.grad,torch.full_like(x,2/x.numel()))
    bigger=p(torch.full((1,1,3,20,40),.2),35)
    assert float(bigger.overhead_bpp)==1
    saved=p.state_dict();assert all(isinstance(v,torch.Tensor) for v in saved.values())
    clone=ContentProxy(Bank(),Codec());clone.load_state_dict(saved)
    assert torch.equal(clone.content_scale,p.content_scale)

def test_scale_fit_does_not_learn_non_vcl_bytes():
    rows=[dict(qp=q,raw_bpp=r,content_bpp=3*r,source='train/a') for q in (30,35,40,45) for r in (.1,.2,.4)]
    assert all(abs(s-3)<1e-6 for s in fit_scales(rows))
    rows[0]['content_bpp']=0
    with pytest.raises(ValueError):fit_scales(rows)

def test_gradient_norms_match_weighted_terms_and_cosine():
    w=torch.nn.Parameter(torch.tensor([1.,2.]))
    info=gradient_balance(w.sum(),2*w.sum(),(w*w).sum(),[w])
    assert info['gradient_rate_norm']==pytest.approx(math.sqrt(2))
    assert info['gradient_task_norm']==pytest.approx(math.sqrt(8))
    assert info['gradient_rate_task_ratio']==pytest.approx(.5)
    assert info['gradient_rate_task_cosine']==pytest.approx(1)

def test_screen_gate_rejects_large_errors_and_direction_failure():
    cfg={'rate_median_error_max':.35,'rate_p90_error_max':.8,'rate_direction_min':.75}
    good=[dict(real_bpp=1.,relaxed_bpp=1.1)]*8
    assert screen_gate(good,dict(aligned=8,untied=8),cfg)['passed']
    assert not screen_gate([dict(real_bpp=1.,relaxed_bpp=8.)]*8,dict(aligned=8,untied=8),cfg)['passed']
    assert not screen_gate(good,dict(aligned=3,untied=8),cfg)['passed']


def test_content_proxy_uses_identity_ste_for_frozen_bank_and_preserves_decoder_grad_when_trainable():
    class DummyBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.3))
        def forward(self, x, qp):
            return SimpleNamespace(reconstruction=self.scale * x, bpp=(x.square().mean() + 0.1))

    class DummyCodec:
        def overhead(self, x, qp):
            return 16

    bank = DummyBank()
    proxy = ContentProxy(bank, DummyCodec())
    x = torch.rand(1, 1, 3, 8, 8)
    out = proxy(x, 35)
    out.reconstruction.sum().backward()
    assert bank.scale.grad is not None and float(bank.scale.grad.abs()) > 0

    bank.requires_grad_(False)
    x_npp = torch.rand(1, 1, 3, 8, 8, requires_grad=True)
    out_frozen = proxy(x_npp, 35)
    assert torch.allclose(out_frozen.reconstruction, 0.3 * x_npp)
    grad_x = torch.autograd.grad(out_frozen.reconstruction.sum(), x_npp)[0]
    assert torch.allclose(grad_x, torch.ones_like(x_npp))


def test_content_proxy_mode_power_stabilizes_high_qp_backward_slope():
    class LowRateBank(torch.nn.Module):
        def forward(self, x, qp):
            return SimpleNamespace(reconstruction=x, bpp=0.05 * x.mean())

    class DummyCodec:
        def overhead(self, x, qp):
            return 16

    bank = LowRateBank().train()
    proxy = ContentProxy(bank, DummyCodec())
    proxy.separate_modes = True
    proxy.relaxed_scale.copy_(torch.tensor([1.0, 1.2, 1.5, 2.0]))
    proxy.relaxed_power.copy_(torch.tensor([1.0, 1.4, 2.0, 3.0]))

    x = torch.full((1, 1, 3, 8, 8), 1.0, requires_grad=True)
    out = proxy(x, 45)
    # Forward value must equal exact power law: 2.0 * (0.05 ** 3.0)
    expected_content = 2.0 * (0.05 ** 3.0)
    assert float(out.content_bpp) == pytest.approx(expected_content, rel=1e-5)

    # Backward gradient wrt raw_bpp must be floored at >= 0.70 * scale instead of vanishing (3 * 0.05^2 = 0.0075 * scale)
    grad_x = torch.autograd.grad(out.bpp, x)[0]
    raw_grad = 0.05 / x.numel()
    assert float(grad_x.mean()) >= 0.70 * 2.0 * raw_grad - 1e-9

