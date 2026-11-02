import copy
import json
from pathlib import Path
import pytest
import torch
from reproduce.residual_npp import ResidualPaperNPP
from reproduce.sam_residual_study import config, ARMS, paired_bd_rate_bootstrap
from reproduce.sam_residual_cloud import prepare, verify
from reproduce.coco_codec_study import validate_warm_start


def test_residual_npp_starts_above_45db_and_has_active_gradients_in_all_branches():
    torch.manual_seed(20261101)
    npp = ResidualPaperNPP(adaptive=False).train()
    x = torch.rand(1, 1, 3, 32, 32)
    y = npp(x, 35)
    mse = float((y - x).square().mean().detach())
    psnr = -10.0 * torch.log10(torch.tensor(mse).clamp_min(1e-12)).item()
    assert psnr > 45.0, f"expected >45 dB initial PSNR, got {psnr:.2f} dB"

    w = torch.linspace(0.5, 1.5, x.numel()).reshape_as(x)
    loss = (y * w).mean()
    loss.backward()
    assert npp.network.conv1.weight.grad is not None
    assert float(npp.network.conv1.weight.grad.abs().sum()) > 0
    assert npp.network.UNet.d1.Conv_BN_ReLU_2[0].weight.grad is not None
    assert float(npp.network.UNet.d1.Conv_BN_ReLU_2[0].weight.grad.abs().sum()) > 0


def test_residual_npp_from_fixed_preserves_exact_forward_and_allows_negative_residuals():
    torch.manual_seed(7)
    fixed = ResidualPaperNPP(adaptive=False).eval()
    x = torch.full((1, 1, 3, 32, 32), 0.6)
    with torch.no_grad():
        fixed.network.conv2.bias.fill_(-0.5)
    y_fixed = fixed(x, 35)
    assert float((y_fixed - x).mean()) < 0.0, "residual NPP must allow negative pixel adjustments"
    adaptive = ResidualPaperNPP.from_fixed(fixed).eval()
    with torch.no_grad():
        assert torch.allclose(fixed(x, 35), adaptive(x, 35), atol=1e-6)


def test_residual_three_arm_config_and_warmstart_validation():
    saved_cfg = config('is', 'control', 'h264')
    saved_cfg['runtime_identity'] = {'train': 'same'}
    saved = dict(
        config=saved_cfg,
        source_sha256='s',
        optimizer_phase='warmup',
        state=dict(task_updates=0, warmup_converged=True),
        npp={'a': 1},
        proxy={'a': 1},
        optimizer={'a': 1},
        rng={'a': 1},
    )
    for arm in ARMS:
        target = config('is', arm, 'h264')
        target['runtime_identity'] = {'train': 'same'}
        validate_warm_start(saved, target, 's')
        assert target['npp_architecture'] == 'residual_tanh_v1'
        assert target['rate_profile'] == 'nal_content_v2_power'
        assert target['gate_psnr'] == 32.0 and target['collapse_psnr'] == 24.0
        assert not target['identity_fallback']
    assert config('is', 'foreground')['boundary_weight'] == 0.0
    assert config('is', 'sam_boundary')['boundary_weight'] == 2.0


def test_residual_cloud_payload_binds_runner_and_rejects_tamper(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for task, codec in (('is', 'h264'), ('od', 'h265')):
        p = tmp_path / (task + codec)
        prepare(root, p, 'testowner', 'sam-residual', config(task, 'control', codec))
        _, job = verify(p, root)
        code = ''.join(json.loads((p / 'notebook.ipynb').read_text())['cells'][0]['source'])
        assert "'reproduce.sam_residual_study'" in code
        assert job['config']['npp_architecture'] == 'residual_tanh_v1'
        tampered = copy.deepcopy(job)
        tampered['config']['gate_psnr'] = 10.0
        (p / 'job.json').write_text(json.dumps(tampered))
        with pytest.raises(ValueError):
            verify(p, root)


def test_paired_bd_rate_bootstrap_reports_ci_and_valid_fraction():
    anchor_curves = {
        30: [{'source': f'img/{i}', 'bytes': 1000 + i * 10, 'pixels': 10000, 'quality': 0.50} for i in range(8)],
        35: [{'source': f'img/{i}', 'bytes': 600 + i * 6, 'pixels': 10000, 'quality': 0.45} for i in range(8)],
        40: [{'source': f'img/{i}', 'bytes': 350 + i * 4, 'pixels': 10000, 'quality': 0.38} for i in range(8)],
        45: [{'source': f'img/{i}', 'bytes': 200 + i * 2, 'pixels': 10000, 'quality': 0.30} for i in range(8)],
    }
    cand_curves = {
        q: [{**pt, 'bytes': int(pt['bytes'] * 0.85)} for pt in pts]
        for q, pts in anchor_curves.items()
    }
    res = paired_bd_rate_bootstrap(anchor_curves, cand_curves, n_boot=50, seed=20261101)
    assert res['valid_draws'] == 50
    assert res['mean_percent'] == pytest.approx(-15.0, abs=0.5)
    assert res['ci95_low'] <= res['mean_percent'] <= res['ci95_high']


def test_adaptive_residual_npp_suppresses_background_highfreq_and_preserves_edges():
    torch.manual_seed(20261101)
    fixed = ResidualPaperNPP(adaptive=False).eval()
    npp = ResidualPaperNPP.from_fixed(fixed).eval()
    h, w = 64, 64
    x = torch.full((1, 1, 3, h, w), 0.5)
    grid = ((torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 2).float() * 2.0 - 1.0
    x[:, :, :, :32, :] += 0.04 * grid[:32, :]
    x[:, :, :, 40:56, 20:44] = 0.85
    x = x.clamp(0.0, 1.0)
    with torch.no_grad():
        y = npp(x, 35)
    bg_hf_before = (x[:, :, :, :28, 1:] - x[:, :, :, :28, :-1]).square().mean()
    bg_hf_after = (y[:, :, :, :28, 1:] - y[:, :, :, :28, :-1]).square().mean()
    edge_before = (x[:, :, :, 40, 22:42] - x[:, :, :, 39, 22:42]).abs().mean()
    edge_after = (y[:, :, :, 40, 22:42] - y[:, :, :, 39, 22:42]).abs().mean()
    edge3_before = (x[:, :, :, 41, 22:42] - x[:, :, :, 38, 22:42]).abs().mean()
    edge3_after = (y[:, :, :, 41, 22:42] - y[:, :, :, 38, 22:42]).abs().mean()
    mse = float((y - x).square().mean())
    psnr = -10.0 * torch.log10(torch.tensor(mse).clamp_min(1e-12)).item()
    assert bg_hf_after < 0.12 * bg_hf_before, "background high-frequency energy must be strongly suppressed (<0.12x)"
    assert 0.72 * edge_before <= edge_after <= 0.95 * edge_before, "1-px edge jump must be preserved without Nyquist overshoot"
    assert edge3_after >= 0.94 * edge3_before, "3-px structural boundary contrast must be preserved"
    assert psnr > 30.0, f"expected >30 dB PSNR on structured scene, got {psnr:.2f} dB"

    # Verify QP monotonicity of background suppression even under adversarial freq_gate values
    with torch.no_grad():
        npp.freq_gate.copy_(torch.tensor([[-5.0, -5.0, 5.0], [5.0, 5.0, -5.0], [-5.0, -5.0, 5.0], [5.0, 5.0, -5.0]]))
        bg_energies = []
        for qp in (30, 35, 40, 45):
            yq = npp(x, qp)
            bg_energies.append(float((yq[:, :, :, :28, 1:] - yq[:, :, :, :28, :-1]).square().mean()))
    assert bg_energies[0] >= bg_energies[1] >= bg_energies[2] >= bg_energies[3]
    assert max(bg_energies) < 0.12 * float(bg_hf_before)

    # Verify all adaptive branches receive finite non-zero gradients
    npp.train()
    npp.zero_grad(set_to_none=True)
    y_train = npp(x, 40)
    w_loss = torch.linspace(0.2, 1.8, x.numel()).reshape_as(x)
    (y_train * w_loss).mean().backward()
    assert npp.freq_gate.grad is not None and float(npp.freq_gate.grad.abs().sum()) > 0
    assert npp.network.conv1.weight.grad is not None and float(npp.network.conv1.weight.grad.abs().sum()) > 0
    assert npp.network.UNet.d1.Conv_BN_ReLU_2[0].weight.grad is not None and float(npp.network.UNet.d1.Conv_BN_ReLU_2[0].weight.grad.abs().sum()) > 0


