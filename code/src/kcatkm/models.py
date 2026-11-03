"""Model registry. Every model maps X (n, D) -> Y (n, 2) in TARGETS order.

Config shape:
    model:
      name: lgbm
      params: {n_estimators: 800, learning_rate: 0.03}
      per_target:                 # optional overrides for one target
        log10_km: {num_leaves: 31}

Two kinds of models:
  * single-target sklearn regressors ("ridge", "lgbm", ...) are wrapped in
    PerTarget, which fits one copy per target;
  * joint models (e.g. a future multi-task NN) register a class with
    fit(X, Y) / predict(X) -> (n, 2) directly via @register_joint.
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import TARGETS


def _ridge(alpha=10.0, **kw):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha, **kw))


def _lgbm(**kw):
    from lightgbm import LGBMRegressor
    kw = {"verbose": -1, "n_jobs": -1, **kw}
    return LGBMRegressor(**kw)


def _xgb(**kw):
    from xgboost import XGBRegressor
    kw = {"tree_method": "hist", "n_jobs": -1, "verbosity": 0, **kw}
    return XGBRegressor(**kw)


def _catboost(**kw):
    from catboost import CatBoostRegressor
    kw = {"verbose": 0, "thread_count": -1, "allow_writing_files": False, **kw}
    return CatBoostRegressor(**kw)


def _hgb(**kw):
    return HistGradientBoostingRegressor(**kw)


def _extratrees(**kw):
    kw = {"n_jobs": -1, **kw}
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(**kw))


SINGLE = {
    "mean": lambda **kw: DummyRegressor(strategy="mean"),
    "ridge": _ridge,
    "lgbm": _lgbm,
    "xgb": _xgb,
    "catboost": _catboost,
    "hgb": _hgb,
    "extratrees": _extratrees,
}
JOINT = {}


def register_joint(name):
    def deco(cls):
        JOINT[name] = cls
        return cls
    return deco


class PerTarget:
    """One regressor per target. target_transform:
      none        fit on the logged values as given
      rank_gauss  fit on the targets mapped to normal scores by rank (the metric
                  is rank-based; this also tames extreme labels), then map
                  predictions back to the log scale so RMSE stays meaningful
    """

    def __init__(self, name, params=None, per_target=None, seed=42, target_transform="none"):
        if target_transform not in ("none", "rank_gauss"):
            raise ValueError(f"unknown target_transform {target_transform!r}")
        self.name, self.params, self.per_target, self.seed = name, params or {}, per_target or {}, seed
        self.target_transform = target_transform
        self.models, self.transforms = [], []

    def _make(self, target):
        kw = {**self.params, **self.per_target.get(target, {})}
        if self.name in ("lgbm", "xgb", "hgb", "extratrees", "catboost"):
            kw.setdefault("random_state", self.seed)
        return SINGLE[self.name](**kw)

    def fit(self, X, Y, only=None, sample_weight=None):
        """only: target indices to fit (the others predict 0); used by CrossTarget."""
        from sklearn.preprocessing import QuantileTransformer
        self.models, self.transforms = [], []
        for j, t in enumerate(TARGETS):
            if only is not None and j not in only:
                self.models.append(None)
                self.transforms.append(None)
                continue
            y = Y[:, j]
            ok = np.isfinite(y)  # a missing target (kcat-only extra rows) is skipped for that target only
            Xj, swj = X, sample_weight
            if not ok.all():
                Xj, y = X[ok], y[ok]
                swj = None if sample_weight is None else np.asarray(sample_weight)[ok]
            tf = None
            if self.target_transform == "rank_gauss":
                tf = QuantileTransformer(n_quantiles=min(1000, len(y)), output_distribution="normal",
                                         random_state=self.seed).fit(y.reshape(-1, 1))
                y = tf.transform(y.reshape(-1, 1)).ravel()
            m = self._make(t)
            if swj is None:
                m.fit(Xj, y)
            elif hasattr(m, "steps"):  # sklearn Pipeline: the weight goes to its last step
                m.fit(Xj, y, **{f"{m.steps[-1][0]}__sample_weight": swj})
            else:
                m.fit(Xj, y, sample_weight=swj)
            self.models.append(m)
            self.transforms.append(tf)
        return self

    def predict(self, X):
        cols = []
        for m, tf in zip(self.models, getattr(self, "transforms", [None] * len(self.models))):
            if m is None:
                cols.append(np.zeros(len(X)))
                continue
            if self.name == "catboost":  # CatBoost ignores OMP_NUM_THREADS; the grader has 2 vCPUs
                import os
                p = m.predict(X, thread_count=int(os.environ.get("OMP_NUM_THREADS", -1)))
            else:
                p = m.predict(X)
            if tf is not None:
                p = tf.inverse_transform(p.reshape(-1, 1)).ravel()
            cols.append(p)
        return np.column_stack(cols)


class Ensemble:
    """Weighted average of several models trained on the same feature matrix.

        model:
          name: ensemble
          members:
            - {name: lgbm, weight: 0.5, params: {...}, target_transform: none}
            - {name: mlp,  weight: {log10_kcat: 0.6, log10_km: 0.2}, params: {...}}

    Weights are normalised per target (scripts/tune_weights.py fits them from OOFs).

    Averaging happens on the log scale. Member predictions are kept by
    `member_predictions` so experiment.py can save them for blend tuning.
    """

    def __init__(self, members, seed=42):
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.specs = [dict(m) for m in members]
        # weight: a number (same for both targets) or {log10_kcat: w, log10_km: w}
        W = np.array([[float(m.get("weight", 1.0)[t]) for t in TARGETS] if isinstance(m.get("weight"), dict)
                      else [float(m.get("weight", 1.0))] * len(TARGETS) for m in self.specs])
        if (W < 0).any() or (W.sum(0) <= 0).any():
            raise ValueError("ensemble weights must be >= 0 with a positive sum per target")
        self.weights = W / W.sum(0)  # shape (n_members, n_targets)
        self.seed = seed
        self.members = []

    needs_groups = True  # passed on to members that use them (CrossTarget)

    def set_feature_names(self, names):
        self.feature_names = list(names)

    def fit(self, X, Y, groups=None, sample_weight=None):
        self.members = []
        for i, spec in enumerate(self.specs):
            if not self.weights[i].any():  # weight 0 for every target: skip (saves fit and inference time)
                self.members.append(None)
                continue
            # seed_offset keeps a member's seed when runs are merged (scripts/merge_runs.py)
            off = int(spec.get("seed_offset", i))
            spec = {k: v for k, v in spec.items() if k not in ("weight", "seed_offset")}
            m = make_model(spec, seed=self.seed + off)
            if hasattr(m, "set_feature_names") and getattr(self, "feature_names", None) is not None:
                m.set_feature_names(self.feature_names)
            self.members.append(fit_model(m, X, Y, groups, sample_weight))
        return self

    def member_predictions(self, X):
        return [np.zeros((len(X), len(TARGETS))) if m is None else np.asarray(m.predict(X), dtype=float)
                for m in self.members]

    def predict(self, X):
        return sum(w * p for w, p in zip(self.weights, self.member_predictions(X)))


class CrossTarget:
    """Feed one target's prediction to the other target's model (kcat and Km are correlated,
    and Km is predicted much better).

        model:
          name: cross
          base: {name: lgbm, params: {...}}       # stage 1, predicts both targets
          stage2: {...}                           # optional, defaults to base
          sources: {log10_kcat: [log10_km]}       # target <- predictions it gets as extra columns
          inner_folds: 3

    Training: the source predictions given to stage 2 are OUT-OF-FOLD (inner GroupKFold on the
    enzyme clusters, like the label features), so stage 2 sees predictions as noisy as at test
    time. Prediction: stage 1 (fitted on all rows) predicts; stage 2 gets those predictions.
    Targets without sources keep stage 1's prediction.
    """
    needs_groups = True

    def __init__(self, base, stage2=None, sources=None, inner_folds=3, seed=42):
        self.base, self.stage2_spec = dict(base), dict(stage2 or base)
        self.sources = sources or {"log10_kcat": ["log10_km"]}
        for t, src in self.sources.items():
            if t not in TARGETS or any(x not in TARGETS or x == t for x in src):
                raise ValueError(f"bad cross sources {self.sources!r}")
        self.inner_folds, self.seed = int(inner_folds), seed
        self.src_idx = sorted({TARGETS.index(x) for v in self.sources.values() for x in v})

    def _fit(self, spec, X, Y, only, groups=None, seed=None, sample_weight=None):
        m = make_model(spec, seed=self.seed if seed is None else seed)
        if isinstance(m, PerTarget):
            return m.fit(X, Y, only=only, sample_weight=sample_weight)
        return fit_model(m, X, Y, groups, sample_weight)

    def _extra(self, t, S):
        return S[:, [TARGETS.index(x) for x in self.sources[t]]]

    def fit(self, X, Y, groups=None, sample_weight=None):
        from sklearn.model_selection import GroupKFold, KFold
        if not np.isfinite(Y).all():
            raise ValueError("cross models need both targets on every row (no kcat-only extra data)")
        sw = None if sample_weight is None else np.asarray(sample_weight, float)
        X = np.asarray(X, dtype=float)
        n = len(X)
        if groups is None or len(np.unique(groups)) < self.inner_folds:
            splitter = KFold(self.inner_folds, shuffle=True, random_state=self.seed).split(X)
        else:
            splitter = GroupKFold(self.inner_folds, shuffle=True, random_state=self.seed).split(X, groups=groups)
        S = np.zeros((n, len(TARGETS)))
        for k, (tr, va) in enumerate(splitter):
            g = None if groups is None else np.asarray(groups)[tr]
            S[va] = self._fit(self.base, X[tr], Y[tr], self.src_idx, g, seed=self.seed + 100 + k,
                              sample_weight=None if sw is None else sw[tr]).predict(X[va])
        self.stage1 = self._fit(self.base, X, Y, None, groups, sample_weight=sw)
        self.stage2 = {}
        for t in self.sources:
            j = TARGETS.index(t)
            self.stage2[t] = self._fit(self.stage2_spec, np.hstack([X, self._extra(t, S)]), Y, [j], groups,
                                       seed=self.seed + 7, sample_weight=sw)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        P = np.asarray(self.stage1.predict(X), dtype=float).copy()
        S = P.copy()  # stage 2 always gets stage-1 predictions, never other stage-2 outputs
        for t, m in self.stage2.items():
            j = TARGETS.index(t)
            P[:, j] = m.predict(np.hstack([X, self._extra(t, S)]))[:, j]
        return P


def fit_model(model, X, Y, groups=None, sample_weight=None):
    kw = {} if sample_weight is None else {"sample_weight": sample_weight}
    if getattr(model, "needs_groups", False):
        return model.fit(X, Y, groups=groups, **kw)
    return model.fit(X, Y, **kw)


def make_model(spec: dict, seed: int = 42):
    name = spec["name"]
    if name == "ensemble":
        return Ensemble(spec["members"], seed=seed)
    if name == "cross":
        return CrossTarget(spec["base"], spec.get("stage2"), spec.get("sources"), spec.get("inner_folds", 3),
                           seed=seed)
    if name in JOINT:
        return JOINT[name](**spec.get("params", {}), seed=seed)
    if name in SINGLE:
        return PerTarget(name, spec.get("params"), spec.get("per_target"), seed=seed,
                         target_transform=spec.get("target_transform", "none"))
    raise KeyError(f"unknown model {name!r}; available: {sorted(SINGLE) + sorted(JOINT) + ['ensemble', 'cross']}")


from . import nn  # noqa: E402,F401  (registers neural joint models; torch is imported lazily)
