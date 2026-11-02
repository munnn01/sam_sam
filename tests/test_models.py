import importlib.util
import pytest
import torch
torch.set_num_threads(2)

def model_api():
    assert importlib.util.find_spec('reproduce.models') is not None, 'paper model wrappers not implemented'
    return __import__('reproduce.models',fromlist=['*'])

def codec_api():
    assert importlib.util.find_spec('reproduce.virtual_codec') is not None, 'structured virtual codec not implemented'
    return __import__('reproduce.virtual_codec',fromlist=['*'])

def test_image_paper_npp_preserves_rgb_geometry_and_gradients():
    # ReLU/clamp in the official model can legitimately saturate arbitrary random
    # initializations. Test a reproducible state; cloud preflight rejects dead runs.
    torch.manual_seed(20261023)
    m=model_api().ImageNPP(adaptive=False).eval()
    x=torch.rand(1,1,3,35,41,requires_grad=True)
    out=m(x,35)
    assert out.shape==x.shape and out.min()>=0 and out.max()<=1
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_adaptive_phase_preserves_the_fixed_npp_at_transition():
    a=model_api()
    fixed=a.ImageNPP(adaptive=False).eval(); adaptive=a.ImageNPP.from_fixed(fixed).eval()
    x=torch.rand(1,1,3,32,48)
    assert torch.allclose(fixed(x,35),adaptive(x,35),atol=1e-7)

def test_video_npp_has_temporal_context_and_resets_per_clip():
    a=model_api(); torch.manual_seed(11)
    m=a.VideoNPP(width=16).eval()
    x=torch.rand(1,4,3,16,16); y=x.clone(); y[:,0]=0
    assert m(x,35).shape==x.shape
    assert not torch.allclose(m(x,35)[:,1],m(y,35)[:,1])
    assert torch.equal(m(x,35),m(x,35))

def test_block_transform_is_invertible_on_non_square_frames():
    c=codec_api(); x=torch.rand(2,3,16,24)
    assert torch.allclose(c.block_idct(c.block_dct(x,8),8),x,atol=2e-6)

def test_virtual_codec_exposes_real_residual_entropy_gradients():
    c=codec_api(); m=c.VirtualCodec(block=8,motion_radius=4)
    x=torch.rand(1,3,3,16,24,requires_grad=True)
    out=m(x,35)
    assert out.reconstruction.shape==x.shape
    assert out.bpp.ndim==0 and torch.isfinite(out.bpp) and out.bpp>0
    (out.bpp+10*(out.reconstruction-x.detach()).square().mean()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_virtual_codec_reference_is_local_to_each_video():
    c=codec_api(); m=c.VirtualCodec(); x=torch.rand(1,2,3,16,16)
    before=m(x,30)
    m(torch.rand_like(x),45)
    after=m(x,30)
    assert torch.equal(before.reconstruction,after.reconstruction)
    assert torch.equal(before.bpp,after.bpp)

def test_virtual_codec_rejects_undeclared_qp():
    c=codec_api()
    with pytest.raises(ValueError): c.VirtualCodec()(torch.rand(1,2,3,16,16),50)
