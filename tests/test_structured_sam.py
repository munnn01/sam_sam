import pytest
import torch
from reproduce.evaluation import processed_video
from reproduce.sam_structured import structured_sam_maps,StructuredMaskCache,SAMGaussianNPP

def settings():
 return dict(min_area_fraction=.005,max_area_fraction=.6,pred_iou_thresh=.8,stability_score_thresh=.9,max_masks=8,smoothing=5)
def test_instance_seams_and_two_channel_geometry():
 a=torch.zeros(12,12);b=a.clone();a[2:10,1:6]=1;b[2:10,6:11]=1
 rows=[dict(segmentation=m,predicted_iou=1.,stability_score=1.) for m in (a,b)]
 maps,info=structured_sam_maps(rows,(12,12),settings())
 assert maps.shape==(1,1,2,12,12) and maps[0,0,1,5,5]>0 and maps[0,0,1,5,6]>0
 assert info['accepted']==2 and info['edge_coverage']>0
 empty,_=structured_sam_maps([], (12,12),settings())
 assert empty.sum()==0

def test_structured_cache_rejects_tampered_edge(tmp_path):
 p=tmp_path/'x.bin';p.write_bytes(b'image')
 row=dict(id='image/1',path=str(p));x=torch.rand(1,1,3,12,12)
 cache=StructuredMaskCache(tmp_path/'cache',settings());maps=torch.rand(1,1,2,12,12)
 cache.store(row,x,maps)
 assert cache.load(row,x).shape==maps.shape
 saved=torch.load(cache.path(row),weights_only=False);saved['mask'][0,0,1,0,0]=.123
 torch.save(saved,cache.path(row))
 with pytest.raises(ValueError):cache.load(row,x)

def test_gaussian_preserves_constant_and_changes_texture():
 g=SAMGaussianNPP(.8);constant=torch.full((1,1,3,16,18),.4)
 assert torch.allclose(g(constant,35),constant,atol=1e-6)
 assert torch.allclose(processed_video(g,constant,35,1),constant,atol=1e-6)
 torch.manual_seed(19);x=torch.rand_like(constant)
 assert g(x,35).std()<x.std() and g(x,35).shape==x.shape


def test_structured_objective_distills_post_codec_edges_and_fpn_features():
    from torch.nn import functional as F
    from reproduce.sam_structured import structured_objective
    from reproduce.sam_residual_study import config
    from reproduce import tri_task_study as study

    cfg = config('od', 'sam_boundary', 'h264')
    x = torch.rand(1, 1, 3, 32, 32)
    y = (x + 0.02 * torch.randn_like(x)).requires_grad_(True)
    decoded = y + (x - 0.05 - y).detach()
    rate = y.square().mean() * 0.5
    task = (decoded - x).square().mean()
    maps = torch.zeros(1, 1, 2, 32, 32)
    maps[:, :, 0, 8:24, 8:24] = 0.9
    maps[:, :, 1, 8:10, 8:24] = 1.0

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        def forward(self, z):
            return {'0': self.conv(z), '1': F.avg_pool2d(self.conv(z), 2)}

    class DummyTaskObj:
        def __init__(self):
            self.model = type('M', (), {'backbone': DummyBackbone().eval()})()

    study._active_decoded = decoded
    study._active_task_obj = DummyTaskObj()
    try:
        loss, parts = structured_objective(cfg, rate, task, x, y, 35, maps)
        assert 'post_codec_boundary' in parts
        assert 'fpn_feature_loss' in parts
        assert 'lowfreq_drift' in parts
        assert 'bg_highpass_y' in parts
        assert float(parts['post_codec_boundary']) > 0.0
        assert float(parts['fpn_feature_loss']) > 0.0
        assert float(parts['bg_highpass_y']) > 0.0
        grad_y = torch.autograd.grad(loss, y, retain_graph=True)[0]
        assert torch.isfinite(grad_y).all() and float(grad_y.abs().sum()) > 0.0

        # Verify QP-balanced task/distillation scaling between QP 30 and QP 45
        loss_30, _ = structured_objective(cfg, rate * 0.0, task, x, y, 30, maps)
        loss_45, _ = structured_objective(cfg, rate * 0.0, task, x, y, 45, maps)
        g30 = float(torch.autograd.grad(loss_30, y, retain_graph=True)[0].norm())
        g45 = float(torch.autograd.grad(loss_45, y, retain_graph=True)[0].norm())
        assert g45 / max(g30, 1e-9) < 3.5, f"expected QP 45 vs QP 30 task/distill gradient ratio < 3.5, got {g45 / max(g30, 1e-9):.2f}"
    finally:
        study._active_decoded = None
        study._active_task_obj = None
