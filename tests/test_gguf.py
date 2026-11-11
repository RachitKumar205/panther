import re

import gguf
import numpy as np
import pytest

from panther.gguf_files import TensorStore, assemble, verify
from panther.model import ModelGeometry, build_inventory, gene_bytes

TINY = ModelGeometry(
    name="tiny", n_layers=12, hidden=256, intermediate=512, n_heads=4, n_kv_heads=2,
    vocab=64, tied=False, max_context=128,
)


def _write_source(inv, fmt: str, path, seed: int = 0):
    """A uniform GGUF of the tiny model in `fmt`, built the way llama-quantize --pure would."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, "qwen2")
    w.add_uint32("qwen2.block_count", inv.geometry.n_layers)
    w.add_string("general.name", f"tiny-{fmt}")
    qt = gguf.GGMLQuantizationType[fmt]
    for g in inv.genes:
        for t in g.tensors:
            data = rng.standard_normal((t.rows, t.cols)).astype(np.float32)
            w.add_tensor(t.name, gguf.quants.quantize(data, qt), raw_dtype=qt)
    for t in inv.fixed_tensors:
        w.add_tensor(t.name, np.ones((t.cols,), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture
def store(tmp_path):
    inv = build_inventory(TINY)
    srcs = {f: _write_source(inv, f, tmp_path / f"{f}.gguf", seed=i) for i, f in enumerate(["Q8_0", "Q4_0"])}
    return inv, TensorStore(srcs)


def test_assemble_and_verify_mixed(store, tmp_path):
    inv, ts = store
    a = {g.gene_id: ("Q4_0" if g.layer is not None and g.layer % 2 else "Q8_0") for g in inv.genes}
    out = assemble(inv, a, ts, tmp_path / "mixed.gguf", base_format="Q8_0")
    rep = verify(inv, a, out)
    assert rep.ok, rep.mismatches[:3]

    mixed = {t.name: t for t in gguf.GGUFReader(out).tensors}
    srcs = {f: {t.name: t for t in gguf.GGUFReader(p).tensors} for f, p in ts.sources.items()}
    for g in inv.genes:
        for t in g.tensors:
            assert np.array_equal(mixed[t.name].data, srcs[a[g.gene_id]][t.name].data)
    # Norms/biases carried over; metadata copied.
    assert "output_norm.weight" in mixed
    r = gguf.GGUFReader(out)
    assert r.fields["qwen2.block_count"].contents() == TINY.n_layers


def test_predicted_bytes_match_file(store, tmp_path):
    inv, ts = store
    a = {g.gene_id: "Q4_0" for g in inv.genes}
    out = assemble(inv, a, ts, tmp_path / "q4.gguf")
    actual = sum(t.n_bytes for t in gguf.GGUFReader(out).tensors if t.name in {
        s.name for g in inv.genes for s in g.tensors})
    assert actual == sum(gene_bytes(g, "Q4_0") for g in inv.genes)


def test_verify_flags_mismatch(store, tmp_path):
    inv, ts = store
    a = {g.gene_id: "Q8_0" for g in inv.genes}
    out = assemble(inv, a, ts, tmp_path / "q8.gguf")
    wrong = dict(a, **{"mlp.3": "Q4_0"})
    rep = verify(inv, wrong, out)
    assert not rep.ok and {m[0] for m in rep.mismatches} == {
        "blk.3.ffn_gate.weight", "blk.3.ffn_up.weight", "blk.3.ffn_down.weight"}
