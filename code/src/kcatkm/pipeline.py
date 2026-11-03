"""Features + model as one object. The same object is used in CV, for the
full-data fit, and (pickled) inside the submission image, so there is no
train/serve skew.

Optional `label_features` (see labelfeat.py) are fitted on the training rows
passed to fit_xy and appended to X before the model sees it. They need `meta`:
a frame with smiles, sequence and (for training rows) the CV `group`.
"""
import platform

import joblib
import numpy as np
import pandas as pd

from . import TARGETS, __version__
from .features import build_matrix, make_extractors
from .models import make_model


class KcatKmPipeline:
    def __init__(self, feature_specs, model_spec, seed: int = 42, clip=(-15.0, 15.0), label_features=None,
                 sample_weight=None):
        self.feature_specs = feature_specs
        self.model_spec = model_spec
        self.label_features = label_features
        # sample_weight: {power: p}: each row weighs 1 / (rows of its enzyme)^p, so heavily measured
        # enzymes do not dominate training (the test set has ~2 rows per enzyme). p=0: unweighted.
        self.sample_weight = sample_weight
        self.seed = seed
        self.clip = clip  # targets live roughly in [-8, 8]; guards against wild extrapolation
        self.extractors = make_extractors(feature_specs)
        self.model = None
        self.lf = None
        self.feature_names = None
        self.meta = {}

    def features(self, df, use_cache=True):
        X, names = build_matrix(df, self.extractors, use_cache=use_cache)
        self.feature_names = names
        return X

    @staticmethod
    def make_meta(df, groups=None):
        meta = df[["smiles", "sequence"]].reset_index(drop=True).copy()
        if groups is not None:
            meta["group"] = np.asarray(groups)
        return meta

    def output_feature_names(self):
        extra = self.lf.names() if getattr(self, "lf", None) is not None else []
        return list(self.feature_names or []) + extra

    def augment(self, X, meta=None):
        """X plus label-derived features (no-op without label_features)."""
        if getattr(self, "lf", None) is None:
            return X
        if meta is None:
            raise ValueError("this pipeline uses label_features: pass meta (smiles, sequence)")
        return np.hstack([X, self.lf.transform(X, meta)])

    def fit_xy(self, X, Y, meta=None, names=None):
        if names is not None:
            self.feature_names = list(names)
        if self.label_features:
            from .labelfeat import LabelFeatures
            if meta is None:
                raise ValueError("label_features need meta (smiles, sequence, group)")
            # rows with a missing target (kcat-only extra rows) are label references for the other target
            self.lf = LabelFeatures(self.label_features, self.feature_names, seed=self.seed).fit(X, Y, meta)
            F = self.lf.train_features_
            X = np.hstack([X, F])
            del self.lf.train_features_  # out-of-fold training features are not needed after fitting
        groups = None
        if meta is not None:
            groups = meta["group"].to_numpy() if "group" in meta else pd.factorize(meta["sequence"])[0]
        from .models import fit_model
        sw = None
        spec = getattr(self, "sample_weight", None)
        if spec and float(spec.get("power", 0)) > 0:
            if meta is None:
                raise ValueError("sample_weight needs meta (sequence)")
            counts = meta["sequence"].map(meta["sequence"].value_counts()).to_numpy(float)
            sw = counts ** -float(spec["power"])
            sw = sw / sw.mean()
        model = make_model(self.model_spec, seed=self.seed)
        if hasattr(model, "set_feature_names"):  # models that group columns by name (fusion)
            model.set_feature_names(self.output_feature_names())
        self.model = fit_model(model, X, Y, groups, sw)
        return self

    def fit(self, df, use_cache=True, groups=None):
        X = self.features(df, use_cache)
        return self.fit_xy(X, df[list(TARGETS)].to_numpy(float), self.make_meta(df, groups))

    def _finish(self, P):
        P = np.asarray(P, dtype=float)
        P = np.nan_to_num(P, nan=0.0, posinf=self.clip[1], neginf=self.clip[0])
        return np.clip(P, *self.clip)

    def predict_x(self, X, meta=None):
        return self._finish(self.model.predict(self.augment(X, meta)))

    def member_predictions_x(self, X, meta=None):
        """Per-member predictions for ensemble models (empty list otherwise)."""
        if not hasattr(self.model, "member_predictions"):
            return []
        return [self._finish(p) for p in self.model.member_predictions(self.augment(X, meta))]

    def predict(self, df, use_cache=False):
        return self.predict_x(self.features(df, use_cache), self.make_meta(df))

    def save(self, path):
        import sklearn
        self.meta.update(kcatkm=__version__, python=platform.python_version(), sklearn=sklearn.__version__)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "KcatKmPipeline":
        return joblib.load(path)
