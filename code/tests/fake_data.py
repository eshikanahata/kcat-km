"""Synthetic data with the real schema (enzyme families + real SMILES), for tests."""
import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
SMILES = ["CCO", "CC(=O)O", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "c1ccccc1O", "NC(Cc1ccccc1)C(=O)O",
          "CC(C)CC(N)C(=O)O", "O=C(O)CCC(=O)O", "BrCCBr", "C#N", "CCCCCCCCCCCCCCCC(=O)O",
          "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O", "not_a_smiles(", "CC(=O)C(=O)O", "OCC(O)CO"]


def make(n_train=1500, n_val=200, n_families=60, seed=0):
    rng = np.random.default_rng(seed)
    fams = ["".join(rng.choice(list(AA), rng.integers(80, 400))) for _ in range(n_families)]
    fam_effect = rng.normal(0, 1.0, (n_families, 2))
    smi_effect = rng.normal(0, 0.7, (len(SMILES), 2))

    def rows(n, fam_ids, prefix):
        out = []
        for i in range(n):
            f = rng.choice(fam_ids)
            s = list(fams[f])
            for p in rng.choice(len(s), max(1, len(s) // 20), replace=False):
                s[p] = rng.choice(list(AA))
            j = rng.integers(len(SMILES))
            y = fam_effect[f] + smi_effect[j] + 0.3 * np.log(len(s)) + rng.normal(0, 0.4, 2)
            out.append({"id": f"{prefix}_{i + 1:06d}", "smiles": SMILES[j], "sequence": "".join(s),
                        "log10_kcat": y[0], "log10_km": y[1]})
        return pd.DataFrame(out)

    train_f = np.arange(int(n_families * 0.8))
    val_f = np.arange(int(n_families * 0.8), n_families)  # unseen families, like the real val
    train = rows(n_train, train_f, "tr")
    val = rows(n_val, val_f, "v")[["id", "smiles", "sequence"]]
    return train, val


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tr, va = make()
    tr.to_parquet(out / "train.parquet", index=False)
    va.to_parquet(out / "val_inputs.parquet", index=False)
    print(f"wrote fake data to {out}")
