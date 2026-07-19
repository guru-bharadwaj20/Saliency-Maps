import torch

from detection.fusion_module import SaliencyFusion


def test_output_shape():
    fusion = SaliencyFusion(use_attention=True)
    rgb = torch.rand(2, 3, 64, 64)
    sal = torch.rand(2, 1, 64, 64)
    out = fusion(rgb, sal)
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
    fusion = SaliencyFusion(use_attention=False)
    fusion.eval()
    wild_sal = torch.full((1, 1, 32, 32), 10.0)  # out-of-range saliency
    out = fusion(torch.rand(1, 3, 32, 32), wild_sal)
    assert torch.isfinite(out).all()
