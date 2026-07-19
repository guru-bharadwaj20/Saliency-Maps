import torch

from saliency_driving.fusion import SaliencyFusion


def test_output_shape():
    fusion = SaliencyFusion(use_attention=True)
    out = fusion(torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 3, 64, 64)


def test_no_attention_variant():
    fusion = SaliencyFusion(use_attention=False)
    out = fusion(torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_gradients_flow():
    fusion = SaliencyFusion(use_attention=True)
    out = fusion(torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    out.sum().backward()
    grads = [p.grad for p in fusion.parameters()]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_saliency_clamped():
    fusion = SaliencyFusion(use_attention=False).eval()
    out = fusion(torch.rand(1, 3, 32, 32), torch.full((1, 1, 32, 32), 10.0))
    assert torch.isfinite(out).all()
