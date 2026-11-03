"""Import extra kcat / Km training data (e.g. the Leak-CURBER kinetic_params_dataset from
Harvard Dataverse, doi:10.7910/DVN/TEKPCS) into data/raw/extra_train.parquet.

    python scripts/import_extra.py PATH --inspect      # what is in there (files, columns, values)
    python scripts/import_extra.py PATH                # import (auto-detects columns and one split folder)
    python scripts/import_extra.py PATH --split threshold_0.3 --smiles-col SMILES --seq-col Sequence

PATH is a folder (searched recursively for .parquet / .csv) or a single file.

What it does
  1. Finds the kcat and the Km tables (from folder / file / column names).
  2. Every split folder of the benchmark (train/val/test of one threshold) holds the SAME
     records partitioned differently, so only ONE split folder per target is used (else
     every record would be counted several times). --split picks it; default: the one
     with the most rows.
  3. Labels follow the competition's rules: per (SMILES, sequence) pair, kcat = max and
     Km = mean of the raw values, then log10. Values that are already log10 (column name
     contains "log", or negative values present) are converted back first.
  4. Keeps pairs that have BOTH targets (the models predict both).
  5. CHECKS THE CONVENTIONS: on pairs that are also in the competition's training data,
     the imported labels must agree with the competition's (Spearman ~1, offset ~0). A
     constant offset means different units: fix it with --kcat-offset / --km-offset.
  6. Writes data/raw/extra_train.parquet (id, smiles, sequence, log10_kcat, log10_km),
     INCLUDING pairs already in train (they are dropped when training, see extra.py).
Then use it with  data: {extra_data: true}  in a config.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _bootstrap  # noqa: F401
from kcatkm import data, extra


def read(f: Path) -> pd.DataFrame:
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)


def files_under(path: Path):
    if path.is_file():
        return [path]
    return sorted([*path.rglob("*.parquet"), *path.rglob("*.csv")])


def find_col(cols, explicit, include, exclude=(), exact=()):
    if explicit:
        if explicit not in cols:
            raise SystemExit(f"column {explicit!r} not found; columns: {list(cols)}")
        return explicit
    low = {c: c.lower() for c in cols}
    for c in cols:
        if low[c] in exact:
            return c
    hits = [c for c in cols if any(s in low[c] for s in include) and not any(s in low[c] for s in exclude)]
    return hits[0] if len(hits) == 1 else (None if not hits else hits)


def target_of(f: Path, root: Path):
    parts = [p.lower() for p in f.relative_to(root).parts] if f != root else [f.name.lower()]
    text = "/".join(parts)
    has_kcat = "kcat" in text and "kcat_km" not in text and "kcatkm" not in text
    has_km = any(p in ("km",) or p.startswith("km_") or p.startswith("km.") or p.endswith("_km")
                 for p in parts) or "/km/" in "/" + text + "/"
    if has_kcat and not has_km:
        return "kcat"
    if has_km and not has_kcat:
        return "km"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--split", help="substring selecting ONE split folder per target (e.g. threshold_0.3)")
    ap.add_argument("--smiles-col")
    ap.add_argument("--seq-col")
    ap.add_argument("--kcat-col")
    ap.add_argument("--km-col")
    ap.add_argument("--kcat-offset", type=float, default=0.0, help="added to log10 kcat (unit fix)")
    ap.add_argument("--km-offset", type=float, default=0.0, help="added to log10 Km (unit fix)")
    args = ap.parse_args()

    root = Path(args.path).expanduser()
    fs = files_under(root)
    if not fs:
        raise SystemExit(f"no .parquet/.csv files under {root}")
    base = root if root.is_dir() else root.parent

    if args.inspect:
        for f in fs[:60]:
            d = read(f)
            print(f"{f.relative_to(base)}  {d.shape}  target={target_of(f, base)}")
            print("   columns:", list(d.columns))
        f = fs[0]
        print(f"\nfirst rows of {f.relative_to(base)}:")
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            print(read(f).head(3).to_string())
        if len(fs) > 60:
            print(f"... {len(fs) - 60} more files")
        return

    # ---- collect per target: files grouped by their folder; keep ONE folder per target
    by_target = {"kcat": {}, "km": {}}
    for f in fs:
        t = target_of(f, base)
        if t is None:
            cols = list(read(f).columns)
            low = [c.lower() for c in cols]
            if any("kcat" in c and "km" not in c for c in low) and any(c in ("km", "log10_km") for c in low):
                t = "both"
            else:
                continue
        if args.split and args.split not in str(f.relative_to(base)):
            continue
        for tt in (("kcat", "km") if t == "both" else (t,)):
            by_target[tt].setdefault(f.parent, []).append(f)

    tables = {}
    for t, folders in by_target.items():
        if not folders:
            print(f"no {t} files found" + (f" matching --split {args.split!r}" if args.split else ""))
            continue
        sizes = {d: sum(len(read(f)) for f in fl) for d, fl in folders.items()}
        chosen = max(sizes, key=sizes.get)
        if len(folders) > 1:
            print(f"{t}: {len(folders)} split folders found; using {chosen.relative_to(base)} ({sizes[chosen]} rows; "
                  "others are re-partitions of the same records). Choose another with --split.")
        d = pd.concat([read(f) for f in folders[chosen]], ignore_index=True)
        cols = d.columns
        smi = find_col(cols, args.smiles_col, ["smiles"], exact=("smiles",))
        seq = find_col(cols, args.seq_col, ["sequence", "seq"], exclude=["id", "len"],
                       exact=("sequence", "seq", "protein_sequence"))
        val = find_col(cols, args.kcat_col if t == "kcat" else args.km_col,
                       ["kcat"] if t == "kcat" else ["km"],
                       exclude=["km"] if t == "kcat" else ["kcat"],
                       exact=("kcat", "log10_kcat") if t == "kcat" else ("km", "log10_km"))
        if val is None:
            val = find_col(cols, None, ["label", "target", "value"], exact=("y", "label", "target", "value"))
        for name, c in (("SMILES", smi), ("sequence", seq), (f"{t} value", val)):
            if not isinstance(c, str):
                raise SystemExit(f"{t}: cannot tell which column is the {name} ({c}); columns: {list(cols)}. "
                                 f"Pass it explicitly, e.g. --{'smiles' if name == 'SMILES' else 'seq' if name == 'sequence' else t}-col NAME")
        v = pd.to_numeric(d[val], errors="coerce")
        is_log = "log" in val.lower() or (v.dropna() <= 0).any()
        raw = 10.0 ** v if is_log else v
        x = pd.DataFrame({"smiles": d[smi].astype(str), "sequence": d[seq].astype(str), "raw": raw})
        x = x[np.isfinite(x["raw"]) & (x["raw"] > 0)]
        agg = x.groupby(["smiles", "sequence"])["raw"].agg("max" if t == "kcat" else "mean")
        off = args.kcat_offset if t == "kcat" else args.km_offset
        tables[t] = (np.log10(agg) + off).rename(f"log10_{t}")
        print(f"{t}: {len(d)} rows from {len(folders[chosen])} files, columns smiles={smi!r} sequence={seq!r} "
              f"value={val!r} ({'log10' if is_log else 'raw'} values) -> {len(agg)} pairs "
              f"({'max' if t == 'kcat' else 'mean'} of repeats)")

    if set(tables) != {"kcat", "km"}:
        raise SystemExit("need both a kcat and a Km table")
    both = pd.concat([tables["kcat"], tables["km"]], axis=1, join="inner").reset_index()
    print(f"pairs with both kcat and Km: {len(both)}")

    # ---- convention check against the competition's training labels
    train = data.load_train()
    m = both.merge(train[["smiles", "sequence", "log10_kcat", "log10_km"]], on=["smiles", "sequence"],
                   suffixes=("", "_train"))
    if len(m) >= 20:
        print(f"\nconvention check on {len(m)} pairs that are also in the competition's training data:")
        bad = False
        for t in ("kcat", "km"):
            a, b = m[f"log10_{t}"], m[f"log10_{t}_train"]
            r = spearmanr(a, b).statistic
            off = float(np.median(b - a))
            same = float(np.mean(np.abs(a - b) < 0.01))
            print(f"  {t}: Spearman {r:.3f}, median offset (train - imported) {off:+.3f}, identical {same:.0%}")
            if r < 0.9 or abs(off) > 0.2:
                bad = True
        if bad:
            print("  !! the imported labels do NOT match the competition's. Do not train on them until this is "
                  "understood (a constant offset = units: rerun with --kcat-offset/--km-offset).")
    else:
        print(f"only {len(m)} pairs overlap the training data: conventions cannot be checked")

    pairs = set(zip(train["smiles"], train["sequence"]))
    new = np.array([p not in pairs for p in zip(both["smiles"], both["sequence"])])
    new_enz = ~both["sequence"].isin(set(train["sequence"]))
    val_inputs = data.load_val_inputs()
    in_val = both.set_index(["smiles", "sequence"]).index.isin(
        val_inputs.set_index(["smiles", "sequence"]).index)
    print(f"\nnew pairs (not in train): {new.sum()} | of those with an enzyme not in train: {(new & new_enz).sum()} "
          f"| pairs also in the public val inputs: {in_val.sum()}")

    out = both.copy()
    out.insert(0, "id", [f"lc_{i:06d}" for i in range(len(out))])
    extra.EXTRA_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(extra.EXTRA_FILE, index=False)
    print(f"wrote {extra.EXTRA_FILE} ({len(out)} pairs; {new.sum()} will be used: pairs already in train are "
          "skipped at training time)")
    print("next: compute embeddings for the new enzymes (Colab: python scripts/pack_colab.py, then gpu_embed), "
          "then run a config with  data: {extra_data: true}")


if __name__ == "__main__":
    main()
