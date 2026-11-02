import importlib.util
import pytest
import torch
from reproduce.virtual_codec import VirtualCodec

torch.set_num_threads(2)

def api():
    assert importlib.util.find_spec('reproduce.proxy_variants') is not None, 'diagnostic codec not implemented'
    return __import__('reproduce.proxy_variants',fromlist=['*'])

def test_diagnostic_ste_is_identical_to_original_codec():
    a=api(); torch.manual_seed(12)
    original=VirtualCodec(); diagnostic=a.DiagnosticCodec.from_legacy(original)
    x=torch.rand(1,3,3,16,24)
    old=original(x,35); new=diagnostic(x,35)
    assert torch.equal(old.reconstruction,new.reconstruction)
    assert torch.equal(old.bpp,new.bpp)
    assert torch.allclose(new.bpp,new.entropy_bpp+new.mode_bpp)
    assert torch.allclose(new.excess_bpp,new.entropy_bpp-new.zero_reference_bpp)

def test_reference_cost_handles_padding_and_first_frame_mode():
    a=api(); codec=a.DiagnosticCodec()
    x=torch.full((1,2,3,9,10),.5)
    out=codec(x,45)
    expected=-codec.selected_prior(45)(torch.zeros(1,192,1,1)).log2().sum()*4/(9*10)
    assert float(out.zero_reference_bpp)==pytest.approx(float(expected),rel=1e-6)
    assert float(out.excess_bpp)==pytest.approx(0,abs=3e-6)
    assert float(out.mode_bpp)==pytest.approx(4*torch.log2(torch.tensor(10.)).item()/(2*9*10))

def test_uniform_rate_changes_gradient_without_changing_task_reconstruction():
    a=api(); codec=a.DiagnosticCodec().requires_grad_(False)
    x=torch.full((1,2,3,8,8),.51,requires_grad=True)
    ste=codec(x,45,rate_mode='ste')
    noise=codec(x,45,rate_mode='uniform',generator=torch.Generator().manual_seed(4))
    assert torch.equal(ste.reconstruction,noise.reconstruction)
    g1,=torch.autograd.grad(ste.bpp,x)
    g2,=torch.autograd.grad(noise.bpp,x)
    assert torch.isfinite(g2).all() and not torch.allclose(g1,g2)

def test_per_qp_prior_copies_are_independent_and_start_identically():
    a=api(); original=VirtualCodec(); bank=a.DiagnosticCodec.from_legacy(original,per_qp=True)
    assert bank.selected_prior(30) is not bank.selected_prior(45)
    before=bank.selected_prior(45).matrices[0].clone()
    with torch.no_grad():bank.selected_prior(30).matrices[0].add_(1)
    assert torch.equal(before,bank.selected_prior(45).matrices[0])

def test_uniform_probe_resolves_latent_position_inside_zero_bin():
    a=api(); model=VirtualCodec().prior
    rows=a.quantizer_probe(model,positions=(0.,.25),samples=2048)
    assert rows[0]['ste_gradient_abs']<1e-6
    assert rows[0]['noise_expected_gradient_abs']<1e-6
    assert rows[1]['ste_gradient_abs']<1e-6
    assert rows[1]['noise_expected_gradient_abs']>.1
