import json

import numpy as np
import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from csa_adapter.longform.data import groups, split_calls
from csa_adapter.longform.evaluate import windows
from csa_adapter.longform.memory import MemoryBank, MemoryConfig, PersistentCSA
from csa_adapter.longform.runtime import adapt_valid, asr_loss, load_adapter, save_adapter
from csa_adapter.longform.train import load_index


def adapter(**kw):
    cfg = dict(
        model_dim=16,
        adapter_dim=8,
        index_dim=4,
        value_dim=4,
        rank=4,
        heads=2,
        compression_rate=2,
        top_k=2,
        max_memory=8,
        query_chunk=3,
        memory_chunk=2,
    )
    cfg.update(kw)
    return PersistentCSA(MemoryConfig(**cfg))


def test_empty_history_and_padding_unchanged():
    a = adapter()
    h = torch.randn(1, 8, 16)
    assert torch.equal(a(h), h)
    m = a.compress(torch.randn(1, 5, 16))
    out = adapt_valid(a, h, 3, m)
    assert torch.equal(out[:, 3:], h[:, 3:])
    assert not torch.equal(out[:, :3], h[:, :3])


def test_exact_topk_and_sparse_dense_equivalence():
    torch.manual_seed(1)
    a = adapter(top_k=100)
    raw = torch.randn(1, 7, 16)
    mem = a.compress(torch.randn(1, 9, 16))
    sparse = a(raw, mem)
    dense = a(raw, mem, dense=True)
    assert torch.allclose(sparse, dense, atol=1e-6)
    a.cfg.top_k = 2
    q = torch.randn(1, 2, 7, 4)
    k = torch.randn(1, 2, 9, 4)
    expected = (q @ k.transpose(-1, -2)).topk(2, dim=-1).indices
    assert torch.equal(a.select(q, k), expected)


def test_historical_writer_gets_gradient_but_backbone_does_not():
    torch.manual_seed(7)
    a = adapter()
    hist = torch.randn(1, 7, 16, requires_grad=True)
    m = a.historical_memory([hist])
    a(torch.randn(1, 5, 16), m).square().mean().backward()
    assert hist.grad is None
    for name in [
        "compressor.proj.weight",
        "compressor.event_proj.weight",
        "compressor.event_score.weight",
        "q.down.weight",
        "k.down.weight",
    ]:
        grad = dict(a.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_commit_reset_capacity_and_no_cross_call_leakage():
    b = MemoryBank(3)
    b.reset("a")
    z = torch.randn(1, 2, 8, requires_grad=True)
    b.commit("a", "0", 0, 10, z)
    with pytest.raises(ValueError):
        b.commit("a", "0", 0, 10, z)
    with pytest.raises(ValueError):
        b.commit("a", "1", 9, 20, z)
    with pytest.raises(ValueError):
        b.commit("b", "1", 10, 20, z)
    b.commit("a", "1", 10, 20, z)
    assert b.memory.shape == (1, 3, 8) and not b.memory.requires_grad
    b.reset("b")
    assert b.memory is None and not b.committed


def test_split_and_history_order():
    s = split_calls([str(i) for i in range(125)])
    assert [len(s[x]) for x in ["train", "validation", "test"]] == [75, 25, 25]
    assert len(set(s["train"]) & set(s["test"])) == 0
    assert s == split_calls(list(reversed([str(i) for i in range(125)])))
    rows = [
        dict(call_id="a", segment_id="1", start=10, end=20),
        dict(call_id="a", segment_id="0", start=0, end=10),
    ]
    assert groups(rows)["a"][0]["segment_id"] == "0"
    with pytest.raises(ValueError):
        groups(rows + [dict(call_id="a", segment_id="2", start=19, end=30)])


def test_no_long_audio_truncation():
    audio = np.zeros(65 * 16000, dtype=np.float32)
    pieces = list(windows(audio, 30))
    assert [(x[0], x[1]) for x in pieces] == [(0, 30), (30, 60), (60, 65)]
    assert sum(len(x[2]) for x in pieces) == len(audio)


def tiny_whisper():
    cfg = WhisperConfig(
        vocab_size=32,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=8,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    model = WhisperForConditionalGeneration(cfg).eval().requires_grad_(False)
    model.generation_config.lang_to_id = {"<|en|>": 3}
    model.generation_config.task_to_id = {"transcribe": 4}
    model.generation_config.no_timestamps_token_id = 5
    model.generation_config.forced_decoder_ids = None
    return model


def test_real_whisper_decoder_backward_and_generation():
    torch.manual_seed(0)
    model = tiny_whisper()
    a = adapter()
    with torch.no_grad():
        h = model.model.encoder(torch.randn(1, 8, 16)).last_hidden_state
    m = a.historical_memory([h[:, :6]])
    changed = adapt_valid(a, h, 7, m)
    loss = asr_loss(model, changed, torch.tensor([[3, 4, 5, 10, 2]]))
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert a.compressor.event_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        for prompt in (None, torch.tensor([6, 9, 10])):
            kw = {} if prompt is None else dict(prompt_ids=prompt)
            ids = model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=changed),
                language="english",
                task="transcribe",
                return_timestamps=False,
                max_new_tokens=3,
                **kw,
            )
            assert ids.ndim == 2


def test_checkpoint_roundtrip_and_backbone_guard(tmp_path):
    a = adapter()
    identity = {"model_id": "tiny-test"}
    save_adapter(tmp_path, a, identity, {})
    restored = load_adapter(tmp_path, identity, "cpu")
    for x, y in zip(a.parameters(), restored.parameters()):
        assert torch.equal(x, y)
    with pytest.raises(ValueError):
        load_adapter(tmp_path, {"model_id": "other"}, "cpu")
    assert not any("memory" in x for x in restored.state_dict())


def test_missing_cache_shard_fails(tmp_path):
    meta = dict(backbone={"model_id": "x"}, shards=2)
    (tmp_path / "metadata-000.json").write_text(json.dumps(meta))
    (tmp_path / "index-000.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete"):
        load_index(tmp_path, {"model_id": "x"})


def test_pack_never_crosses_omitted_segments_or_large_gaps():
    from csa_adapter.longform.data import pack_segments

    rows = [
        dict(call_id="a", segment_id="0", start=0, end=10, text="a", source_order=0),
        dict(call_id="a", segment_id="1", start=10, end=20, text="b", source_order=1),
        dict(call_id="a", segment_id="3", start=21, end=24, text="d", source_order=3),
        dict(call_id="a", segment_id="4", start=30, end=33, text="e", source_order=4),
    ]
    packed = pack_segments(rows)
    assert [r["text"] for r in packed] == ["a b", "d", "e"]
    assert all(r["end"] - r["start"] <= 28 for r in packed)


def test_prepare_full_audio_and_exclusion_report(tmp_path, monkeypatch):
    import io
    import sys
    from types import SimpleNamespace

    import soundfile as sf
    from huggingface_hub import HfApi

    from csa_adapter.longform import data

    buf = io.BytesIO()
    sf.write(buf, np.zeros(60 * 16000, dtype=np.float32), 16000, format="FLAC")
    full = [
        dict(file_id=str(i), audio={"bytes": buf.getvalue()}, transcription="full reference")
        for i in range(5)
    ]
    metadata = []
    for i in range(5):
        for j, (start, end) in enumerate([(0, 10), (10, 20), (20, 51), (51, 55)]):
            metadata.append(
                dict(
                    file_id=str(i),
                    segment_id=str(j),
                    start_ts=start,
                    end_ts=end,
                    transcription=f"word{j}",
                )
            )
    monkeypatch.setattr(data, "hf_rows", lambda *args: iter(full))
    monkeypatch.setattr(data, "hf_segment_metadata", lambda *args: iter(reversed(metadata)))
    monkeypatch.setattr(
        HfApi, "dataset_info", lambda *args, **kwargs: SimpleNamespace(sha="test-revision")
    )
    monkeypatch.setattr(sys, "argv", ["prepare", "--output", str(tmp_path), "--oversize", "skip"])
    data.prepare()
    report = json.loads((tmp_path / "preparation_report.json").read_text())
    assert report["excluded_segments"] == 5
    assert report["retained_segments"] == 10
    test = data.read_jsonl(tmp_path / "test.full.jsonl")
    assert len(test) == 1 and test[0]["duration"] == 60
    all_segments = sum(
        [data.read_jsonl(tmp_path / f"{s}.jsonl") for s in ["train", "validation", "test"]], []
    )
    assert all(r["duration"] <= 28 for r in all_segments)
    assert all("word2" not in r["text"] for r in all_segments)
    assert any(r["text"] == "word0 word1" for r in all_segments)


def test_projected_parquet_metadata_reader(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    from csa_adapter.longform.data import hf_segment_metadata

    file = tmp_path / "test.parquet"
    pq.write_table(
        pa.table(
            dict(
                file_id=["1"],
                segment_id=["0"],
                transcription=["hello"],
                start_ts=[0.0],
                end_ts=[1.0],
                audio=[b"not a decoded audio object"],
            )
        ),
        file,
    )
    monkeypatch.setattr(HfApi, "list_repo_files", lambda *args, **kwargs: ["chunked/test.parquet"])
    monkeypatch.setattr(HfFileSystem, "open", lambda *args, **kwargs: open(file, "rb"))
    rows = list(hf_segment_metadata("test-revision"))
    assert rows == [
        dict(file_id="1", segment_id="0", transcription="hello", start_ts=0.0, end_ts=1.0)
    ]
