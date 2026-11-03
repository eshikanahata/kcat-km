"""How much kcat / Km data do other public collections add beyond what you already have
(the competition's training data + the imported Leak-CURBER file)?

    python scripts/source_overlap.py --download                 # fetch the sources (~50 MB, once)
    python scripts/source_overlap.py                            # the overlap report
    python scripts/source_overlap.py --write                    # + add the NEW pairs to the extra data
    python scripts/source_overlap.py --write --public-only      # ONLY the new pairs -> extra_public.parquet

Sources (public GitHub, BRENDA + SABIO-RK based):
  catpred_kcat  CatPred kcat set   (23,151 entries, s^-1)   github.com/maranasgroup/CatPred
  catpred_km    CatPred Km set     (41,174 entries, mM)     same repository
  dlkcat_kcat   DLKcat kcat set    (17,010 entries, s^-1)   github.com/SysBioChalmers/DLKcat

Different collections spell the same molecule differently, so pairs are matched on RDKit
canonical SMILES (stereo kept) + exact sequence. Within a source, repeats follow the
competition's rules: kcat = max, Km = mean of the raw values, then log10.

The label check compares each source with the competition's labels on shared pairs; a
source whose labels disagree is reported and not written for that target.

--write keeps the current file as data/raw/extra_train_leakcurber.parquet and writes
data/raw/extra_train.parquet = Leak-CURBER pairs (kcat only: its Km failed the label check)
+ the NEW pairs. A target a row has no (trusted) label for is left empty, and that target's
models skip the row. Use  data.extra_targets: [log10_kcat, log10_km]  to train both targets.
"""
import argparse
import json
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _bootstrap  # noqa: F401
from kcatkm import data, extra, paths

EXTERNAL = paths.DATA / "external"
SOURCES = {
    "catpred_kcat": ("log10_kcat", "https://raw.githubusercontent.com/maranasgroup/CatPred/main/external/DLKcat/"
                                   "DeeplearningApproach/Code/model/kcat-random_trainvaltest_dlkcat.json"),
    "catpred_km": ("log10_km", "https://raw.githubusercontent.com/maranasgroup/CatPred/main/external/DLKcat/"
                               "DeeplearningApproach/Code/model/km-random_trainvaltest_dlkcat.json"),
    "dlkcat_kcat": ("log10_kcat", "https://raw.githubusercontent.com/SysBioChalmers/DLKcat/master/"
                                  "DeeplearningApproach/Data/database/Kcat_combination_0918_wildtype_mutant.json"),
}
_CANON = {}


def canon(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        c = _CANON.get(s)
        if c is None:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
            c = Chem.MolToSmiles(m) if m is not None else f"?{s}"
            _CANON[s] = c
        out.append(c)
    return np.asarray(out, dtype=object)


def load_source(name):
    target, _ = SOURCES[name]
    f = EXTERNAL / f"{name}.json"
    if not f.exists():
        raise SystemExit(f"{f} missing: run with --download first")
    d = pd.DataFrame(json.loads(f.read_text()))
    d["raw"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d[np.isfinite(d["raw"]) & (d["raw"] > 0) & d["Smiles"].astype(bool) & d["Sequence"].astype(bool)]
    d = d.assign(smiles=d["Smiles"].astype(str), sequence=d["Sequence"].astype(str))
    d["key"] = canon(d["smiles"])
    g = d.groupby(["key", "sequence"])
    agg = g["raw"].agg("max" if target == "log10_kcat" else "mean")
    out = pd.DataFrame({target: np.log10(agg)}).join(g["smiles"].first()).reset_index()
    return target, out


def keyed(df):
    return set(zip(canon(df["smiles"]), df["sequence"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--exclude-val", action="store_true",
                    help="drop new pairs that are also in the public val inputs (the competition's held-out data)")
    ap.add_argument("--public-only", action="store_true",
                    help="with --write: write ONLY the new public pairs to data/raw/extra_public.parquet "
                         "(no Leak-CURBER rows; the Leak-CURBER file is left as it is)")
    ap.add_argument("--keep-leakcurber-km", action="store_true",
                    help="keep Leak-CURBER's Km labels in the written file (default: blank them; they failed the "
                         "label check against the competition's Km)")
    ap.add_argument("--leakcurber", help="folder with the Leak-CURBER *_1d.parquet files (default: use the "
                                         "imported data/raw/extra_train.parquet)")
    args = ap.parse_args()

    if args.download:
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        for name, (_, url) in SOURCES.items():
            dest = EXTERNAL / f"{name}.json"
            if not dest.exists():
                print(f"downloading {name} ...", flush=True)
                urllib.request.urlretrieve(url, dest)
            print(f"  {dest} ({dest.stat().st_size / 2**20:.1f} MiB)")

    train, val = data.load_train(), data.load_val_inputs()
    print("canonicalising SMILES (once, ~1 minute) ...", flush=True)
    train_keys, val_keys = keyed(train), keyed(val)
    # Leak-CURBER: per target, every pair it has a label for
    lc = {}
    if args.leakcurber:
        from pathlib import Path
        for t in ("kcat", "km"):
            f = next(Path(args.leakcurber).expanduser().glob(f"{t}_*1d*.parquet"))
            d = pd.read_parquet(f, columns=["smiles", "sequence"])
            lc[f"log10_{t}"] = keyed(d)
    else:
        src = extra.EXTRA_FILE
        if src.exists():
            ex = pd.read_parquet(src)
            for t in ("log10_kcat", "log10_km"):
                lc[t] = keyed(ex[ex[t].notna()])
            print(f"Leak-CURBER pairs from {src.name} (kcat & Km pairs only; pass --leakcurber FOLDER for all of them)")
        else:
            lc = {"log10_kcat": set(), "log10_km": set()}
            print("no Leak-CURBER data found: comparing with the training data only")
    seqs_known = set(train["sequence"])
    if extra.EXTRA_FILE.exists():
        seqs_known |= set(pd.read_parquet(extra.EXTRA_FILE, columns=["sequence"])["sequence"])

    tr = train.assign(key=canon(train["smiles"])).set_index(["key", "sequence"])
    new_by_target = {"log10_kcat": [], "log10_km": []}
    usable = {}
    rows = []
    for name in SOURCES:
        target, d = load_source(name)
        k = list(zip(d["key"], d["sequence"]))
        in_train = np.array([x in train_keys for x in k])
        in_lc = np.array([x in lc.get(target, set()) for x in k])
        in_val = np.array([x in val_keys for x in k])
        new = ~in_train & ~in_lc
        new_enz = new & ~d["sequence"].isin(seqs_known).to_numpy()
        # label convention check against the competition's labels
        m = d[in_train].set_index(["key", "sequence"]).join(tr[[target]], rsuffix="_train", how="inner")
        m = m[~m.index.duplicated()]
        if len(m) >= 20:
            r = spearmanr(m[target], m[f"{target}_train"]).statistic
            off = float(np.median(m[f"{target}_train"] - m[target]))
            same = float(np.mean(np.abs(m[target] - m[f"{target}_train"]) < 0.01))
            ok = r >= 0.9 and abs(off) <= 0.2
        else:
            r, off, same, ok = np.nan, np.nan, np.nan, False
        usable[name] = ok
        rows.append({"source": name, "target": target.replace("log10_", ""), "pairs": len(d),
                     "in_train": int(in_train.sum()), "in_leakcurber": int(in_lc.sum()),
                     "in_val_inputs": int(in_val.sum()), "NEW": int(new.sum()), "NEW_new_enzyme": int(new_enz.sum()),
                     "spearman_vs_train": round(r, 3), "offset": round(off, 3), "identical": round(same, 2),
                     "labels_match": "yes" if ok else "NO"})
        if ok:
            new_by_target[target].append(d[new])

    rep = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print("\n" + rep.to_string(index=False))

    print("\nNEW = pair (canonical SMILES + sequence) in neither the training data nor Leak-CURBER")
    combined = {}
    for t, parts in new_by_target.items():
        if not parts:
            combined[t] = pd.DataFrame(columns=["key", "sequence", "smiles", t])
            continue
        c = pd.concat(parts, ignore_index=True)
        c = c.groupby(["key", "sequence"]).agg({t: "max" if t == "log10_kcat" else "mean", "smiles": "first"}).reset_index()
        combined[t] = c
    both = combined["log10_kcat"].merge(combined["log10_km"], on=["key", "sequence"], how="outer",
                                        suffixes=("", "_km"))
    both["smiles"] = both["smiles"].fillna(both.get("smiles_km"))
    n_k = int(both["log10_kcat"].notna().sum()) if len(both) else 0
    n_m = int(both["log10_km"].notna().sum()) if len(both) else 0
    n_e = both.loc[~both["sequence"].isin(seqs_known), "sequence"].nunique() if len(both) else 0
    print(f"all sources together (only sources whose labels match; duplicates across sources merged): "
          f"{n_k} new kcat pairs, {n_m} new Km pairs, {n_e} enzymes not in train or Leak-CURBER")

    if args.exclude_val and len(both):
        in_val = np.array([k in val_keys for k in zip(both["key"], both["sequence"])])
        both = both[~in_val].reset_index(drop=True)
        print(f"--exclude-val: dropped {int(in_val.sum())} new pairs that are in the public val inputs "
              f"({len(both)} left)")
    if args.write and len(both) and args.public_only:
        spell = dict(zip(canon(train["smiles"]), train["smiles"]))
        add = pd.DataFrame({"smiles": [spell.get(k, s) for k, s in zip(both["key"], both["smiles"])],
                            "sequence": both["sequence"], "log10_kcat": both.get("log10_kcat"),
                            "log10_km": both.get("log10_km")})
        add.insert(0, "id", [f"pub_{i:06d}" for i in range(len(add))])
        add[["log10_kcat", "log10_km"]] = add[["log10_kcat", "log10_km"]].astype(float)
        dest = paths.RAW / "extra_public.parquet"
        add.to_parquet(dest, index=False)
        print(f"wrote {dest}: {len(add)} new public pairs only (no Leak-CURBER rows); use the *_pub configs")
    elif args.write and len(both):
        cur = extra.EXTRA_FILE
        backup = paths.RAW / "extra_train_leakcurber.parquet"
        if cur.exists() and not backup.exists():
            pd.read_parquet(cur).to_parquet(backup, index=False)
        base = pd.read_parquet(backup) if backup.exists() else pd.DataFrame(
            columns=["id", "smiles", "sequence", "log10_kcat", "log10_km"])
        if not args.keep_leakcurber_km and len(base):
            base = base.assign(log10_km=np.nan)
            print("Leak-CURBER Km labels blanked (they disagree with the competition's); its kcat labels kept")
        # write substrates in the training data's spelling when it is the same molecule
        spell = dict(zip(canon(train["smiles"]), train["smiles"]))
        add = pd.DataFrame({"smiles": [spell.get(k, s) for k, s in zip(both["key"], both["smiles"])],
                            "sequence": both["sequence"], "log10_kcat": both.get("log10_kcat"),
                            "log10_km": both.get("log10_km")})
        add.insert(0, "id", [f"pub_{i:06d}" for i in range(len(add))])
        out = pd.concat([base, add], ignore_index=True)
        out[["log10_kcat", "log10_km"]] = out[["log10_kcat", "log10_km"]].astype(float)
        out.to_parquet(cur, index=False)
        print(f"wrote {cur}: {len(base)} Leak-CURBER pairs + {len(add)} new pairs "
              f"(backup of the Leak-CURBER-only file: {backup.name})")
        print("next: embeddings for the new enzymes (python scripts/pack_colab.py, then gpu_embed on Colab), "
              "then rerun the _exk configs with --force")


if __name__ == "__main__":
    main()
