"""Cross-validated experiment: one config in, one run folder + one index row out."""
import time

import numpy as np
import pandas as pd

from . import TARGETS, config as cfglib, data, metrics, paths, splits, tracking
from .pipeline import KcatKmPipeline


def score_labelled_val(cfg, df, X, Y, meta, names, seed, log):
    """Fit on ALL training rows and score the instructor-provided labelled validation set."""
    from . import valset
    vdf, info = valset.load(train=df)
    if vdf.empty:
        log(f"labelled val: no usable rows ({info})")
        return None, None
    pipe = KcatKmPipeline(cfg["features"], cfg["model"], seed=seed, label_features=cfg.get("label_features"),
                          sample_weight=cfg.get("sample_weight"))
    pipe.fit_xy(X, Y, meta, names=names)
    Xv = pipe.features(vdf, use_cache=True)
    P = pipe.predict_x(Xv, KcatKmPipeline.make_meta(vdf))
    s = valset.score(vdf, P)
    log(f"labelled val ({info['usable']} rows; {info['pair_in_train_dropped']} dropped as already in train): "
        + valset.fmt(s))
    pred = vdf[["id"]].copy()
    for j, t in enumerate(TARGETS):
        pred[t] = vdf[t].to_numpy()
        pred[f"pred_{t}"] = P[:, j]
    return s, pred


def run_cv(cfg: dict, force: bool = False, notes: str = "", verbose: bool = True, val: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    paths.ensure_dirs()
    from . import extra as extra_data
    use_extra = extra_data.wanted(cfg)
    # the extra file's content is part of the run's identity (re-importing it means a new run)
    fp = cfglib.fingerprint({**cfg, "_extra_data": extra_data.fingerprint(extra_data.file_of(cfg))} if use_extra else cfg)

    df = data.load_train(cfg.get("data", {}).get("train"))
    data_fp = data.fingerprint(df)  # identifies the training-data version (before any subsampling)

    # skip only if the same config was already run on the SAME training data
    idx = tracking.read_index()
    if not force and not idx.empty and "fingerprint" in idx and "data" in idx:
        same = idx[(idx["fingerprint"] == fp) & (idx["data"] == data_fp)]
        if not same.empty:
            prev = same.iloc[-1]
            log(f"identical config already run on this data as {prev['run_id']} "
                f"(mean_scc={prev['mean_scc']:.4f}). Use --force to rerun.")
            return paths.RUNS / prev["run_id"]
    sub = cfg.get("data", {}).get("subsample")
    if sub:
        df = df.sample(n=min(sub, len(df)), random_state=0).reset_index(drop=True)
    from . import valset
    extra = (cfg.get("data") or {}).get("extra_train")
    vdf_extra = None
    if extra:
        df, vdf_extra = valset.training_data(cfg, base=df)
        log(f"training on {len(df)} rows including {len(vdf_extra)} labelled-val rows "
            "(CV and val are scored separately; each val row is predicted by a model that did not see it)")
    is_val = valset.is_val_row(df["id"])

    folds = splits.get_folds(df, cfg["split"])
    log(f"split {cfg['split']['strategy']}: {splits.describe(folds)}")
    block = None
    if use_extra:  # training-only rows; the original rows keep their folds
        if vdf_extra is not None:
            raise ValueError("use either data.extra_train (labelled val) or data.extra_data, not both")
        df, folds, block = extra_data.attach(df, folds, cfg["split"], extra_data.targets_of(cfg),
                                             extra_data.file_of(cfg))
        log("extra data: " + extra_data.describe(df, block))
        is_val = valset.is_val_row(df["id"])
    is_ex = extra_data.is_extra(df["id"])
    noscore = is_val | is_ex

    seeds = cfg.get("seeds", [42])
    lf_spec = cfg.get("label_features")
    template = KcatKmPipeline(cfg["features"], cfg["model"], label_features=lf_spec)
    t0 = time.time()
    X = template.features(df, use_cache=True)
    t_feat = time.time() - t0
    Y = df[list(TARGETS)].to_numpy(float)
    log(f"features: {X.shape[1]} columns from {template.extractors} ({t_feat:.1f}s)")
    meta = KcatKmPipeline.make_meta(df, folds["group"].to_numpy())  # group = enzyme cluster, for inner folds
    if lf_spec:
        log(f"label features: {[k for k in lf_spec if k not in ('emb_prefix', 'inner_folds')]} "
            f"(fitted inside each training fold, inner folds by enzyme cluster)")
    names_out = None

    oof = np.zeros_like(Y)
    member_oof = {}  # ensemble member index -> OOF predictions
    fold_scores, importances = [], []
    t0 = time.time()
    for k in sorted(f for f in folds["fold"].unique() if f >= 0):
        tr = ((folds["fold"] != k).to_numpy() & ~is_ex).copy()
        va = (folds["fold"] == k).to_numpy()
        if block is not None:  # extra rows unlike every enzyme of this validation fold
            tr[np.where(is_ex)[0]] = ~block[:, k]
        preds = []
        for seed in seeds:
            pipe = KcatKmPipeline(cfg["features"], cfg["model"], seed=seed, label_features=lf_spec,
                                  sample_weight=cfg.get("sample_weight"))
            pipe.fit_xy(X[tr], Y[tr], meta.iloc[tr].reset_index(drop=True), names=template.feature_names)
            m_va = meta.iloc[va].reset_index(drop=True)
            preds.append(pipe.predict_x(X[va], m_va))
            for i, p in enumerate(pipe.member_predictions_x(X[va], m_va)):
                member_oof.setdefault(i, np.zeros_like(Y))[va] += p / len(seeds)
            names_out = pipe.output_feature_names()
            for est in [pipe.model, *getattr(pipe.model, "members", [])]:
                for m in getattr(est, "models", []):
                    imp = getattr(m, "feature_importances_", None)
                    if imp is not None and len(imp) == len(names_out):
                        importances.append(imp)
        oof[va] = np.mean(preds, axis=0)
        sc = va & ~is_val  # CV scores use training rows only, so they stay comparable across runs
        s = metrics.score(Y[sc], oof[sc])
        fold_scores.append(s)
        log(f"  fold {k}: mean_scc={s['mean_scc']:.4f} kcat={s['kcat_scc']:.4f} km={s['km_scc']:.4f}")
    t_fit = time.time() - t0

    overall = metrics.score(Y[~noscore], oof[~noscore])
    for i, p in member_oof.items():
        s = metrics.score(Y[~noscore], p[~noscore])
        name = cfg["model"]["members"][i]["name"]
        overall[f"member{i}_{name}_mean_scc"] = s["mean_scc"]
        log(f"  member {i} ({name}) alone: mean_scc={s['mean_scc']:.4f} kcat={s['kcat_scc']:.4f} km={s['km_scc']:.4f}")
    tl = metrics.testlike(Y[~noscore], oof[~noscore], df["sequence"].to_numpy()[~noscore])
    for k, v in tl.items():
        overall[f"tl_{k}"] = v
    tl_folds = []
    for f in sorted(x for x in np.unique(folds["fold"]) if x >= 0):
        m = (folds["fold"].to_numpy() == f) & ~noscore
        tl_folds.append(metrics.testlike(Y[m], oof[m], df["sequence"].to_numpy()[m])["mean_scc"])
    for s_, v in zip(fold_scores, tl_folds):
        s_["tl_mean_scc"] = v
    log(f"test-like score (at most 2 rows per enzyme, like the test set): mean_scc={tl['mean_scc']:.4f} "
        f"kcat={tl['kcat_scc']:.4f} km={tl['km_scc']:.4f}")
    fold_df = pd.DataFrame(fold_scores)
    overall["mean_scc_fold_std"] = float(fold_df["mean_scc"].std())
    log(f"OOF mean_scc={overall['mean_scc']:.4f} (fold std {overall['mean_scc_fold_std']:.4f}) "
        f"kcat={overall['kcat_scc']:.4f} km={overall['km_scc']:.4f} mean_rmse={overall['mean_rmse']:.4f}")

    val_metrics, val_pred = None, None
    if vdf_extra is not None:  # val rows were trained on: score their out-of-fold predictions
        val_metrics = valset.score(vdf_extra, oof[is_val])
        log(f"labelled val (out-of-fold, {is_val.sum()} rows): {valset.fmt(val_metrics)}")
        val_pred = vdf_extra[["id"]].copy()
        for j, t in enumerate(TARGETS):
            val_pred[t] = vdf_extra[t].to_numpy()
            val_pred[f"pred_{t}"] = oof[is_val, j]
    elif val and valset.available() and not cfg.get("data", {}).get("subsample"):
        log(f"labelled val: refitting on all {len(df)} rows and scoring (no output until done; "
            f"~{t_fit / max(len(fold_scores), 1) * 1.3 / 60:.0f} min)...")
        t1 = time.time()
        val_metrics, val_pred = score_labelled_val(cfg, df, X, Y, meta, template.feature_names, seeds[0], log)
        log(f"  (full-data fit + labelled-val scoring took {time.time() - t1:.0f}s)")

    run_id, d = tracking.new_run_dir(cfg["name"])
    tracking.save_yaml(cfg, d / "config.yaml")
    if val_pred is not None:
        val_pred.to_parquet(d / "val_labelled_predictions.parquet", index=False)
    tracking.save_json({"overall": overall, "folds": fold_scores, "fingerprint": fp, "labelled_val": val_metrics,
                        "n_rows": len(df), "n_features": X.shape[1],
                        "feature_seconds": t_feat, "fit_seconds": t_fit}, d / "metrics.json")
    out = pd.DataFrame({"id": df["id"], "fold": folds["fold"], "group": folds["group"]})
    for j, t in enumerate(TARGETS):
        out[t] = Y[:, j]
        out[f"pred_{t}"] = oof[:, j]
        for i, p in member_oof.items():
            out[f"member{i}_pred_{t}"] = p[:, j]
    out = out[~is_ex]  # extra rows are training-only: the OOF file keeps exactly the original rows
    out.to_parquet(d / "oof.parquet", index=False)
    if importances:
        names = names_out
        imp = pd.Series(np.mean(importances, axis=0), index=names).sort_values(ascending=False)
        imp.rename("importance").to_csv(d / "feature_importance.csv")

    tracking.append_index({
        "run_id": run_id, "name": cfg["name"], "fingerprint": fp, "git": tracking.git_commit(),
        "split": splits.label(cfg["split"]), "data": data_fp,
        "extra_train": extra or (f"extra_data:{int(is_ex.sum())}:{extra_data.fingerprint(extra_data.file_of(cfg))}" if use_extra else ""),
        "features": "+".join(e.key.rsplit("_", 1)[0] for e in template.extractors),
        "model": cfg["model"]["name"], "n_features": X.shape[1], "seeds": len(seeds),
        **{k: round(v, 5) for k, v in overall.items() if not k.startswith("member")},
        **({} if val_metrics is None else {
            "val_mean_scc": round(val_metrics["mean_scc"], 5), "val_kcat_scc": round(val_metrics["kcat_scc"], 5),
            "val_km_scc": round(val_metrics["km_scc"], 5),
            "val_unseen_mean_scc": round(val_metrics["unseen_enzyme_mean_scc"], 5)}),
        "fit_seconds": round(t_fit, 1), "notes": notes or cfg.get("notes", ""),
    })
    log(f"saved {d.relative_to(paths.ROOT)}")
    return d
