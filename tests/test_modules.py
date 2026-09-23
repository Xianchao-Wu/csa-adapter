import torch

from csa_adapter import CSAAdapter, CSAAdapterConfig


def make_config(**updates):
    values = dict(
        model_dim=32,
        adapter_dim=16,
        index_dim=8,
        value_dim=8,
        rank=4,
        num_heads=2,
        compression_rate=4,
        top_k=3,
        query_chunk_size=5,
        memory_chunk_size=2,
    )
    values.update(updates)
    return CSAAdapterConfig(**values)


def test_shape_identity_and_gradient_flow():
    module = CSAAdapter(make_config())
    x = torch.randn(2, 13, 32, requires_grad=True)
    correction = module(x)
    assert correction.shape == x.shape
    assert torch.count_nonzero(correction) == 0  # ReZero initialization
    (x + correction).square().mean().backward()
    assert module.alpha.grad is not None
    assert torch.isfinite(module.alpha.grad)


def test_padding_mask_and_routing():
    module = CSAAdapter(make_config())
    module.alpha.data.fill_(1.0)
    x = torch.randn(2, 11, 32)
    mask = torch.tensor([[1] * 11, [1] * 7 + [0] * 4], dtype=torch.bool)
    correction, routing = module(x, padding_mask=mask, return_routing=True)
    assert routing.indices.shape == (2, 2, 11, 3)
    assert torch.allclose(correction[1, 7:], torch.zeros_like(correction[1, 7:]))
    sums = routing.probabilities.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_causal_routing_has_no_future_windows():
    cfg = make_config(causal=True, compression_rate=2, top_k=2)
    module = CSAAdapter(cfg)
    x = torch.randn(1, 10, 32)
    _, routing = module(x, return_routing=True)
    query_pos = torch.arange(10).view(1, 1, 10, 1)
    selected_end = routing.indices * 2 + 1
    assert torch.all((selected_end <= query_pos) | ~routing.valid)


def test_context_bank():
    cfg = make_config(use_context_bank=True, context_dim=24)
    module = CSAAdapter(cfg)
    x = torch.randn(2, 9, 32)
    context = torch.randn(2, 3, 24)
    output = module(x, context_bank=context)
    assert output.shape == x.shape
