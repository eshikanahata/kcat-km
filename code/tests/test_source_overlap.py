"""Other public sources: matching on canonical SMILES, the label check, and --write."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_overlap_report_and_write(tmp_path):
    for name in ("src", "configs", "scripts"):
        (tmp_path / name).symlink_to(ROOT / name)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    ext = tmp_path / "data" / "external"
    ext.mkdir()
    seqs = ["MKTAYIAKQRQ" * 5, "GGSSPPWWHH" * 5, "DEDEKRKRNQ" * 5, "LLIIVVFFYY" * 5]
    train = pd.DataFrame({"id": ["a", "b"], "smiles": ["OCC", "c1ccccc1O"], "sequence": seqs[:2],
                          "log10_kcat": [1.0, 2.0], "log10_km": [0.0, -1.0]})
    train.to_parquet(tmp_path / "data" / "raw" / "train.parquet", index=False)
    train[["id", "smiles", "sequence"]].to_parquet(tmp_path / "data" / "raw" / "val_inputs.parquet", index=False)
    lc = pd.DataFrame({"id": ["l1"], "smiles": ["CC(=O)O"], "sequence": [seqs[2]], "log10_kcat": [0.5],
                       "log10_km": [0.1]})
    lc.to_parquet(tmp_path / "data" / "raw" / "extra_train.parquet", index=False)
    rng = np.random.default_rng(0)

    def entry(smi, seq, val, unit):
        return {"Smiles": smi, "Sequence": seq, "Value": str(val), "Unit": unit}
    # 25 shared pairs so the label check runs: same values as "train" (spelled differently: CCO vs OCC)
    big_train = pd.concat([train] + [pd.DataFrame({"id": [f"x{i}"], "smiles": ["C" * (i + 2)], "sequence": [seqs[3]],
                                                   "log10_kcat": [rng.normal()], "log10_km": [rng.normal()]})
                                     for i in range(25)], ignore_index=True)
    big_train.to_parquet(tmp_path / "data" / "raw" / "train.parquet", index=False)
    kc = [entry("CCO", seqs[0], 10.0, "s^(-1)"), entry("CC(=O)O", seqs[2], 3.0, "s^(-1)"),
          entry("CCN", seqs[3], 5.0, "s^(-1)"), entry("CCN", seqs[3], 50.0, "s^(-1)")]  # new pair, max of repeats
    kc += [entry("C" * (i + 2), seqs[3], 10 ** big_train.loc[i + 2, "log10_kcat"], "s^(-1)") for i in range(25)]
    km = [entry("C" * (i + 2), seqs[3], 10 ** (big_train.loc[i + 2, "log10_km"] + 3), "mM") for i in range(25)]
    (ext / "catpred_kcat.json").write_text(json.dumps(kc))
    (ext / "catpred_km.json").write_text(json.dumps(km))  # labels off by 3 orders: must be rejected
    (ext / "dlkcat_kcat.json").write_text(json.dumps(kc[:3]))
    env = {**os.environ, "KCATKM_ROOT": str(tmp_path)}
    # the new pair CCN/seqs[3] is put into the public val inputs: --exclude-val must drop it
    val = pd.read_parquet(tmp_path / "data" / "raw" / "val_inputs.parquet")
    pd.concat([val, pd.DataFrame({"id": ["v9"], "smiles": ["NCC"], "sequence": [seqs[3]]})]).to_parquet(
        tmp_path / "data" / "raw" / "val_inputs.parquet", index=False)
    r = subprocess.run([sys.executable, "scripts/source_overlap.py", "--write", "--public-only", "--exclude-val"],
                       env=env, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and "dropped 1 new pairs" in r.stdout, r.stdout + r.stderr
    val.to_parquet(tmp_path / "data" / "raw" / "val_inputs.parquet", index=False)
    r = subprocess.run([sys.executable, "scripts/source_overlap.py", "--write", "--public-only"], env=env,
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    pub = pd.read_parquet(tmp_path / "data" / "raw" / "extra_public.parquet")
    assert pub["smiles"].tolist() == ["CCN"]  # only the genuinely new pair; no Leak-CURBER rows
    assert pd.read_parquet(tmp_path / "data" / "raw" / "extra_train.parquet")["id"].tolist() == ["l1"]  # untouched
    r = subprocess.run([sys.executable, "scripts/source_overlap.py", "--write"], env=env, cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    row = [ln for ln in out.splitlines() if ln.strip().startswith("catpred_kcat")][0].split()
    assert row[2:7] == ["28", "26", "1", "1", "1"]  # pairs, in_train (CCO=OCC), in Leak-CURBER, in val, NEW
    assert "catpred_km" in out and out.split("catpred_km")[1].split("\n")[0].rstrip().endswith("NO")
    ex = pd.read_parquet(tmp_path / "data" / "raw" / "extra_train.parquet")
    assert len(ex) == 2 and ex["id"].tolist()[0] == "l1"  # Leak-CURBER row kept + 1 new pair
    assert ex.loc[0, "log10_kcat"] == 0.5 and np.isnan(ex.loc[0, "log10_km"])  # its Km failed the check: blanked
    new = ex.iloc[1]
    assert new["smiles"] == "CCN" and abs(new["log10_kcat"] - np.log10(50)) < 1e-9 and np.isnan(new["log10_km"])
    assert (tmp_path / "data" / "raw" / "extra_train_leakcurber.parquet").exists()
