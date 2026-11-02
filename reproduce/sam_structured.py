"""Offline SAM instance interior/edge maps, post-codec boundary fidelity, and FPN feature distillation."""
import gc
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .sam2_masks import MaskCache, tensor_digest, build_teacher, mask_objective
from .protocol import task_lambda
from .residual_npp import _gaussian_kernel_1d, _separable_blur
from .training import config_digest, rng_state, restore_rng, log_event
from .tri_task_data import load
from .tri_task_models import CocoTask
from .coco_codec_study import weighted_task_loss
from . import tri_task_study as study


def structured_sam_maps(rows, size, settings):
    h, w = size
    accepted = []
    for row in rows:
        m = torch.as_tensor(row['segmentation'], dtype=torch.float32)
        if m.shape != (h, w) or not torch.isfinite(m).all() or m.min() < 0 or m.max() > 1:
            raise ValueError('invalid SAM mask')
        iou = float(row['predicted_iou'])
        stability = float(row['stability_score'])
        if not (math.isfinite(iou) and math.isfinite(stability) and 0 <= iou <= 1 and 0 <= stability <= 1):
            raise ValueError('invalid SAM confidence')
        area = float(m.mean())
        if (settings['min_area_fraction'] <= area <= settings['max_area_fraction']
                and iou >= settings['pred_iou_thresh']
                and stability >= settings['stability_score_thresh']):
            accepted.append((iou * stability, m))
    accepted = sorted(accepted, key=lambda v: v[0], reverse=True)[:settings['max_masks']]
    fg = torch.zeros(h, w)
    edge = torch.zeros(h, w)
    for confidence, m in accepted:
        fg = torch.maximum(fg, confidence * m)
        v = m[None, None]
        e = (F.max_pool2d(v, 3, 1, 1) + F.max_pool2d(-v, 3, 1, 1))[0, 0].clamp(0, 1)
        edge = torch.maximum(edge, confidence * e)
    k = settings['smoothing']
    fg = F.avg_pool2d(fg[None, None], k, 1, k // 2)[None].clamp(0, 1)
    edge = edge[None, None, None].clamp(0, 1)
    maps = torch.cat((fg, edge), dim=2)
    return maps, dict(
        accepted=len(accepted),
        fg_coverage=float(fg.mean()),
        edge_coverage=float(edge.mean()),
        empty=not accepted,
    )


class StructuredMaskCache(MaskCache):
    def __init__(self, root, settings):
        super().__init__(root, dict(settings, structured_maps_schema='instance_core_edge_v1'))

    def store(self, row, x, mask):
        if mask.shape != (*x.shape[:2], 2, *x.shape[-2:]) or not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError('aligned two-channel SAM map required')
        mask = mask.detach().cpu().to(torch.float16).contiguous()
        identity = self.identity(row, x)
        torch.save(
            dict(
                identity=identity,
                identity_sha256=config_digest(identity),
                mask=mask,
                mask_sha256=tensor_digest(mask),
            ),
            self.path(row),
        )

    def load(self, row, x):
        value = torch.load(self.path(row), map_location='cpu', weights_only=False)
        m = value['mask']
        identity = self.identity(row, x)
        if (value['identity'] != identity
                or value['identity_sha256'] != config_digest(identity)
                or value['mask_sha256'] != tensor_digest(m)):
            raise ValueError('SAM map cache identity/content differs')
        if m.shape != (*x.shape[:2], 2, *x.shape[-2:]) or not torch.isfinite(m).all() or m.min() < 0 or m.max() > 1:
            raise ValueError('SAM map shape/range differs')
        return m.float()


def prepare_structured_masks(rows, cfg, out, budget):
    control = cfg['sam_arm'] == 'control'
    root = out / 'sam_maps' if control else out.parent / 'shared_sam_maps'
    cache = StructuredMaskCache(root, cfg['sam2'])
    saved = rng_state()
    teacher = None
    records = []
    try:
        for i, row in enumerate(rows):
            budget.require('structured_sam_cache')
            x, _, _ = load(row, cfg)
            if cache.path(row).is_file():
                maps = cache.load(row, x)
                part = dict(cached=True, fg_coverage=float(maps[:, :, 0].mean()), edge_coverage=float(maps[:, :, 1].mean()))
            elif control:
                maps = torch.zeros(*x.shape[:2], 2, *x.shape[-2:])
                part = dict(control=True, fg_coverage=0.0, edge_coverage=0.0)
                cache.store(row, x, maps)
            else:
                if teacher is None:
                    teacher = build_teacher(cfg['sam2'], out)
                rgb = (x[0, 0].permute(1, 2, 0) * 255).round().byte().numpy()
                with torch.inference_mode():
                    generated = teacher.generate(rgb)
                maps, part = structured_sam_maps(generated, x.shape[-2:], cfg['sam2'])
                cache.store(row, x, maps)
            records.append(dict(source=row['id'], pixels_sha256=tensor_digest(x), **part))
            if i == 0 or (i + 1) % 8 == 0:
                log_event(out, 'sam_structured_cache', sources=i + 1, total=len(rows), arm=cfg['sam_arm'])
    finally:
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restore_rng(saved)
    info = dict(
        sources=records,
        root=str(root),
        schema='instance_core_edge_v1',
        rendered_cohort_sha256=config_digest([(r['source'], r['pixels_sha256']) for r in records]),
    )
    (out / 'sam_cache.json').write_text(json.dumps(info, indent=2))
    return cache, info


class SAMTask(CocoTask):
    def loss(self, x, target):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        try:
            parts = self.model([x[0, 0]], [{k: v.to(x.device) for k, v in target.items()}])
            study._active_task_components = {k: float(v.detach()) for k, v in parts.items()}
            study._active_decoded = x
            study._active_task_obj = self
            raw = weighted_task_loss(parts, 3.0 if self.task == 'is' else 1.0)
            return raw / (1.0 + raw.detach())
        finally:
            self.model.eval()


def _fpn_distillation_loss(decoded, x, spatial_w):
    task_obj = getattr(study, '_active_task_obj', None)
    backbone = getattr(getattr(task_obj, 'model', None), 'backbone', None)
    if backbone is None:
        return decoded.new_zeros(())
    dec_in = decoded.flatten(0, 1)
    ref_in = x.flatten(0, 1).detach()
    with torch.no_grad():
        ref_feats = backbone(ref_in)
    dec_feats = backbone(dec_in)
    if not isinstance(dec_feats, dict) or not dec_feats:
        return decoded.new_zeros(())
    w_2d = spatial_w.flatten(0, 1)
    losses = []
    for key, f_dec in dec_feats.items():
        if key not in ref_feats:
            continue
        f_ref = ref_feats[key].detach()
        w_lvl = F.adaptive_avg_pool2d(w_2d, f_dec.shape[-2:])
        norm = f_ref.square().mean().clamp_min(1e-4) * w_lvl.mean().clamp_min(1e-6)
        losses.append(((f_dec - f_ref).square() * w_lvl).mean() / norm)
    if not losses:
        return decoded.new_zeros(())
    return sum(losses) / len(losses)


def structured_objective(cfg, rate, task, x, y, q, maps):
    if maps.shape != (*y.shape[:2], 2, *y.shape[-2:]) or not torch.isfinite(maps).all() or maps.min() < 0 or maps.max() > 1:
        raise ValueError('aligned SAM interior/edge maps required')
    maps = maps.detach().to(device=y.device, dtype=y.dtype)
    fg = maps[:, :, :1]
    edge = maps[:, :, 1:2]
    interior = fg * (1.0 - edge)
    qp_balance = (1.28 / task_lambda(q)) ** 0.65
    loss, parts = mask_objective(cfg, rate, task * qp_balance, x, y, q, interior)

    k_ref = _gaussian_kernel_1d(0.48, 2).to(device=x.device, dtype=x.dtype)
    x_struct = _separable_blur(x.detach().flatten(0, 1), k_ref).reshape_as(x)
    boundary = ((y - x_struct).square() * edge).sum() / (edge.sum() * 3.0).clamp_min(1.0)

    decoded = getattr(study, '_active_decoded', None)
    study._active_decoded = None
    if decoded is None or decoded.shape != y.shape or decoded.device != y.device:
        decoded = y

    gx_dec = F.pad(decoded[..., :, 1:] - decoded[..., :, :-1], (0, 1, 0, 0))
    gx_ref = F.pad(x_struct[..., :, 1:] - x_struct[..., :, :-1], (0, 1, 0, 0))
    gy_dec = F.pad(decoded[..., 1:, :] - decoded[..., :-1, :], (0, 0, 0, 1))
    gy_ref = F.pad(x_struct[..., 1:, :] - x_struct[..., :-1, :], (0, 0, 0, 1))
    grad_err = (gx_dec - gx_ref).square() + (gy_dec - gy_ref).square()

    fg_extra = float(cfg.get('sam2', {}).get('foreground_extra', 0.0))
    b_weight = float(cfg.get('boundary_weight', 0.0))
    spatial_w = 1.0 + fg_extra * fg + (1.0 + b_weight) * edge
    post_codec_boundary = ((grad_err + 0.5 * (decoded - x_struct).square()) * spatial_w).mean() / spatial_w.mean().clamp_min(1e-6)

    y_flat = y.flatten(0, 1)
    y_hp = (y_flat - F.avg_pool2d(F.pad(y_flat, (1, 1, 1, 1), mode='replicate'), 3, stride=1)).reshape_as(y)
    bg_mask = (1.0 - torch.maximum(fg, edge)).clamp(0.0, 1.0)
    bg_highpass_y = (bg_mask * y_hp.square()).mean()

    low_diff = F.avg_pool2d(F.pad((y - x).flatten(0, 1), (4, 4, 4, 4), mode='replicate'), 9, 1).reshape_as(y)
    lowfreq_drift = low_diff.square().mean()

    fpn_loss = _fpn_distillation_loss(decoded, x_struct, spatial_w)

    parts.update(
        boundary_distortion=boundary,
        post_codec_boundary=post_codec_boundary,
        fpn_feature_loss=fpn_loss,
        bg_highpass_y=bg_highpass_y,
        lowfreq_drift=lowfreq_drift,
        sam_foreground_coverage=fg.mean(),
        sam_edge_coverage=edge.mean(),
    )
    for k, v in getattr(study, '_active_task_components', {}).items():
        parts['native_' + k] = rate.new_tensor(v)

    total = (
        loss
        + b_weight * boundary
        + (1.2 + 0.4 * b_weight) * post_codec_boundary
        + (0.4 + 0.25 * fg_extra + 0.25 * b_weight) * fpn_loss
        + (1.5 + 1.0 * fg_extra + 0.5 * b_weight) * bg_highpass_y
        + 25.0 * lowfreq_drift
    )
    return total, parts


class SAMGaussianNPP(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('positive sigma required')
        r = math.ceil(3 * sigma)
        v = torch.arange(-r, r + 1, dtype=torch.float32)
        v = torch.exp(-v.square() / (2 * sigma * sigma))
        v = v / v.sum()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.register_buffer('kernel', (v[:, None] * v[None, :])[None, None].repeat(3, 1, 1, 1))

    def forward(self, x, qp):
        b, t, c, h, w = x.shape
        r = self.kernel.shape[-1] // 2
        return F.conv2d(F.pad(x.reshape(b * t, c, h, w), (r, r, r, r), mode='replicate'), self.kernel, groups=3).reshape_as(x).clamp(0, 1)
