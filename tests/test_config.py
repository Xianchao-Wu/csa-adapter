import pytest

from csa_adapter import CSAAdapterConfig


def test_invalid_top_k():
    with pytest.raises(ValueError):
        CSAAdapterConfig(top_k=0)


def test_roundtrip():
    cfg = CSAAdapterConfig(layer_indices=[-4, -1], compressor="event")
    assert CSAAdapterConfig.from_dict(cfg.to_dict()) == cfg
