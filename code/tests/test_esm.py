"""ESM extractor tests with a tiny randomly initialised ESM model (no download needed)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO.-", "<null_1>", "<mask>"]


def make_tiny_esm(dest: Path):
    from transformers import EsmConfig, EsmModel, EsmTokenizer
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "vocab.txt").write_text("\n".join(VOCAB))
    EsmTokenizer(str(dest / "vocab.txt")).save_pretrained(dest)
    cfg = EsmConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=32, max_position_embeddings=1026, pad_token_id=1, mask_token_id=32,
                    position_embedding_type="rotary", token_dropout=True)
    torch.manual_seed(0)
    EsmModel(cfg, add_pooling_layer=False).save_pretrained(dest)


def make_tiny_roberta(dest: Path):
    import fake_data
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
    dest.mkdir(parents=True, exist_ok=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(fake_data.SMILES * 20, vocab_size=300,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(dest))
    tok = RobertaTokenizerFast(vocab_file=str(dest / "vocab.json"), merges_file=str(dest / "merges.txt"))
    tok.save_pretrained(dest)
    cfg = RobertaConfig(vocab_size=tok.vocab_size, hidden_size=12, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=24, max_position_embeddings=34, pad_token_id=1)
    torch.manual_seed(0)
    RobertaModel(cfg, add_pooling_layer=False).save_pretrained(dest)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("pretrained")
    make_tiny_esm(d / "esm_tiny")
    make_tiny_roberta(d / "chem_tiny")
    os.environ["KCATKM_PRETRAINED"] = str(d)
    yield d
    os.environ.pop("KCATKM_PRETRAINED", None)


def test_windowing_and_padding(tiny):
    from kcatkm.features.plm import ESM
    seqs = ["MKTAYIAKQRQISFVKSHFSRQ", "MSLGAKPFGEKKFIEIKGRRMAYIDEGTGDPILFQHGNPTSSYLWRNIMPHCAGLGRLIAC"]
    whole = ESM(model="x/esm_tiny").compute(seqs)
    assert whole.shape == (2, 16) and np.isfinite(whole).all()
    # batching with padding must not change an embedding
    alone = ESM(model="x/esm_tiny").compute(seqs[:1])
    np.testing.assert_allclose(whole[0], alone[0], atol=1e-4)
    # a window longer than the sequence is a no-op; a short window still gives a finite vector
    np.testing.assert_allclose(ESM(model="x/esm_tiny", window=500).compute(seqs), whole, atol=1e-5)
    assert np.isfinite(ESM(model="x/esm_tiny", window=10, pooling="mean_max").compute(seqs)).all()
    # the pickled extractor must not carry the loaded model
    import pickle
    ex = ESM(model="x/esm_tiny")
    ex.compute(seqs[:1])
    assert "_model" not in pickle.loads(pickle.dumps(ex)).__dict__


def test_esm_time_budget_guard(tiny, monkeypatch, capsys):
    import re
    from kcatkm.features.plm import ESM
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), n)) for n in (900, 600, 400, 300, 120)]
    full = ESM(model="x/esm_tiny").compute(seqs)
    monkeypatch.setenv("KCATKM_ESM_BUDGET", "1e-9")      # impossible budget -> must truncate
    guarded = ESM(model="x/esm_tiny", batch_tokens=500).compute(seqs)  # several batches, like real data
    msg = capsys.readouterr().err
    cap = int(re.search(r"truncating sequences to (\d+) residues", msg).group(1))
    assert cap < 900 and np.isfinite(guarded).all()
    monkeypatch.delenv("KCATKM_ESM_BUDGET")
    np.testing.assert_allclose(guarded, ESM(model="x/esm_tiny", max_residues=cap).compute(seqs), atol=1e-4)
    assert not np.allclose(guarded[0], full[0])            # the long sequence really was shortened
    monkeypatch.setenv("KCATKM_ESM_BUDGET", "1e6")       # generous budget -> identical to no guard
    np.testing.assert_allclose(ESM(model="x/esm_tiny", batch_tokens=500).compute(seqs), full, atol=1e-4)


def test_esm_truncation_quantization_and_stable_cache_key(tiny):
    import hashlib
    import json
    from kcatkm.features.plm import ESM
    # the default extractor's cache key must not change (users have hours of cached embeddings)
    old = {"n": "esm", "v": 1, "p": {"model": "facebook/esm2_t12_35M_UR50D", "layer": -1, "pooling": "mean",
                                     "window": 1022, "max_residues": None}}
    expected = "esm_" + hashlib.sha1(json.dumps(old, sort_keys=True).encode()).hexdigest()[:10]
    assert ESM().key == expected
    assert ESM(num_layers=20).key != expected and ESM(quantize="int8").key != expected

    seqs = ["MKTAYIAKQRQISFVKSHFSRQ", "MSLGAKPFGEKKFIEIKGRRMAYIDEGTGDPILFQHGNPTSSYLWRNIMPHCAGLGRLIAC"]
    full = ESM(model="x/esm_tiny").compute(seqs)
    one = ESM(model="x/esm_tiny", num_layers=1).compute(seqs)
    q = ESM(model="x/esm_tiny", quantize="int8").compute(seqs)
    both = ESM(model="x/esm_tiny", num_layers=1, quantize="int8").compute(seqs)
    for X in (one, q, both):
        assert X.shape == full.shape and np.isfinite(X).all()
    assert not np.allclose(one, full, atol=1e-3)
    assert np.corrcoef(q.ravel(), full.ravel())[0, 1] > 0.95  # int8 is a close approximation
    with pytest.raises(ValueError):
        ESM(model="x/esm_tiny", num_layers=99).compute(seqs)


def test_smiles_transformer(tiny):
    from kcatkm.features.plm import SmilesTransformer
    smi = ["CCO", "c1ccccc1O", "C" * 200, "not_a_smiles(", "Br.Cc1nc(N2NC(c3ccccc3)=NN2c2ccccc2)sc1C"]
    ex = SmilesTransformer(model="x/chem_tiny", batch_size=2)
    X = ex.compute(smi)
    assert X.shape == (5, 12) and np.isfinite(X).all()
    assert len(ex.feature_names()) == 12
    # order / batching independence (long SMILES is truncated to the context, not an error)
    np.testing.assert_allclose(ex.compute(smi[::-1])[::-1], X, atol=1e-4)


def test_rank_gauss_target_transform():
    from kcatkm.models import make_model
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    Y = np.column_stack([X[:, 0] * 3 + rng.normal(size=300), np.exp(X[:, 1])])
    m = make_model({"name": "ridge", "target_transform": "rank_gauss"}).fit(X, Y)
    P = m.predict(X)
    assert P.shape == (300, 2) and np.isfinite(P).all()
    assert P[:, 0].min() >= Y[:, 0].min() - 1e-6 and P[:, 0].max() <= Y[:, 0].max() + 1e-6  # back on the log scale
    assert np.corrcoef(P[:, 0], Y[:, 0])[0, 1] > 0.8


def test_cv_and_submission(tiny, tmp_path):
    import fake_data
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (tmp_path / name).symlink_to(ROOT / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tr, va = fake_data.make(n_train=400, n_val=60)
    tr.to_parquet(raw / "train.parquet", index=False)
    va.to_parquet(raw / "val_inputs.parquet", index=False)
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}
    feats = ("features=[seq_len, morgan, {esm: {model: x/esm_tiny, window: 100}}, "
             "{chemberta: {model: x/chem_tiny}}]")

    def run(*a):
        r = subprocess.run([sys.executable, *a], env=env, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    out = run("scripts/precompute.py", "configs/lgbm_esm.yaml", "--set", feats)
    assert "total feature time" in out and "new inputs cached" in out
    out = run("scripts/run.py", "configs/lgbm_esm.yaml", "--set", feats, "model.params.n_estimators=50")
    assert "OOF mean_scc" in out and "esm(" in out
    # second run with another model reuses cached embeddings (no 'new inputs cached' lines)
    out = run("scripts/run.py", "configs/lgbm_esm.yaml", "--set", feats, "model.name=ridge", "model.params={}",
              "model.target_transform=rank_gauss")
    assert "new inputs cached" not in out
    # submission must bundle the weights; its inference check points only at the bundled copy
    out = run("scripts/build_submission.py", "best")
    assert "bundled esm_tiny" in out and "bundled chem_tiny" in out and "output valid" in out


def test_esm_layer_average(tiny):
    from kcatkm.features.plm import ESM
    seqs = ["MKTAYIAKQRQISFVKSHFSRQ", "MSLGAKPFGEKKFIEIKGRRMAYIDEGTGDPILFQHGNPTSSYLWRNIMPHCAGLGRLIAC"]
    a = ESM(model="x/esm_tiny", layer=1).compute(seqs)
    b = ESM(model="x/esm_tiny", layer=2).compute(seqs)
    avg = ESM(model="x/esm_tiny", layers=[1, 2]).compute(seqs)
    assert np.allclose(avg, (a + b) / 2, atol=1e-5)  # mean pooling is linear
    assert ESM(layers=[4, 6, 8], num_layers=8).key != ESM(num_layers=8).key
    with pytest.raises(ValueError):
        ESM(layers=[9], num_layers=8)


def test_colab_pack_and_gpu_embed_roundtrip(tiny, tmp_path):
    """pack_colab -> (other machine) gpu_embed -> unzip on the laptop -> run.py finds the cache."""
    import subprocess
    import zipfile
    import fake_data
    laptop, colab = tmp_path / "laptop", tmp_path / "colab"
    laptop.mkdir()
    for name in ("src", "configs", "scripts", "submission_template", "requirements-infer.txt"):
        (laptop / name).symlink_to(ROOT / name)
    (laptop / "data" / "raw").mkdir(parents=True)
    tr, va = fake_data.make(n_train=200, n_val=30)
    tr.to_parquet(laptop / "data" / "raw" / "train.parquet", index=False)
    va.to_parquet(laptop / "data" / "raw" / "val_inputs.parquet", index=False)
    cfg = laptop / "tiny_cfg.yaml"
    cfg.write_text("base: configs/base.yaml\nname: tiny\nfeatures:\n  - seq_len\n"
                   "  - esm: {model: x/esm_tiny, pooling: mean_max, num_layers: 2, layers: [1, 2]}\n"
                   "model: {name: ridge}\n")

    def run(root, *a):
        env = {**os.environ, "KCATKM_ROOT": str(root)}
        r = subprocess.run([sys.executable, *a], env=env, cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout + r.stderr

    run(laptop, "scripts/pack_colab.py")
    zipfile.ZipFile(laptop / "kcat-km-colab.zip").extractall(colab)
    (colab / "tiny_cfg.yaml").write_text(cfg.read_text().replace("configs/base.yaml", "configs/base.yaml"))
    out = run(colab, "scripts/gpu_embed.py", "tiny_cfg.yaml", "--zip", "esm_cache.zip", "--allow-cpu")
    assert "1 embedding sets" in out
    zipfile.ZipFile(colab / "esm_cache.zip").extractall(laptop)
    cached = list((laptop / "data" / "cache" / "features").glob("esm_*/X.npy"))
    assert len(cached) == 1
    before = cached[0].stat().st_mtime_ns
    out = run(laptop, "scripts/run.py", "tiny_cfg.yaml", "--no-val")
    assert "OOF mean_scc" in out
    assert cached[0].stat().st_mtime_ns == before  # used as is, nothing recomputed
