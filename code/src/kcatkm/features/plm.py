"""Protein language-model embeddings (ESM-2 via Hugging Face transformers).

Weights are loaded from a local folder, never from the internet, because the
grader runs offline:
    local dev:   pretrained/<model basename>/   (python scripts/get_pretrained.py)
    submission:  /app/pretrained/<basename>/     (copied by build_submission.py;
                 found through the KCATKM_PRETRAINED env var set in the Dockerfile)

Long sequences: ESM-2 accepts at most 1022 residues. Each sequence is cut into
consecutive windows of `window` residues; per-residue embeddings of all windows
are pooled together, so a 3000-residue enzyme is embedded in full.

    features:
      - esm: {model: facebook/esm2_t12_35M_UR50D, pooling: mean}
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .. import paths
from .base import Extractor, register


STATE = {"truncated": False}  # set when the time guard had to shorten sequences in this process


def time_left(reserve=0.0):
    """Seconds until KCATKM_DEADLINE (set by predict.py) minus a reserve; None if no deadline."""
    d = float(os.environ.get("KCATKM_DEADLINE", 0) or 0)
    return None if not d else d - time.time() - reserve


def batch_tokens(default):
    """Inference can lower activation memory with KCATKM_BATCH_TOKENS."""
    return int(os.environ.get("KCATKM_BATCH_TOKENS", 0) or default)


def pretrained_dir(model_id: str) -> Path:
    base = Path(os.environ.get("KCATKM_PRETRAINED", paths.PRETRAINED))
    return base / model_id.split("/")[-1]


@register
class ESM(Extractor):
    name = "esm"
    entity = "sequence"
    chunk_size = 512
    adaptive = True  # computed last at inference; shortens sequences if the deadline is near

    def __init__(self, model="facebook/esm2_t12_35M_UR50D", layer=-1, pooling="mean",
                 window=1022, max_residues=None, batch_tokens=16000, num_layers=None, quantize=None,
                 layers=None):
        """num_layers: keep only the first N transformer layers (faster; embeddings from layer N).
        quantize: "int8" = dynamic int8 Linear layers on CPU (~1.4x faster).
        Both change the embeddings, so training and inference always use the same setting."""
        if pooling not in ("mean", "mean_max"):
            raise ValueError("pooling must be 'mean' or 'mean_max'")
        if quantize not in (None, "int8"):
            raise ValueError("quantize must be null or 'int8'")
        # only result-affecting settings go into params (and so into the cache key)
        super().__init__(model=model, layer=layer, pooling=pooling, window=window, max_residues=max_residues)
        # added only when used, so existing caches (e.g. the 35M embeddings) keep their keys
        if num_layers:
            self.params["num_layers"] = int(num_layers)
        if quantize:
            self.params["quantize"] = quantize
        if layers:  # average the hidden states of these layers (1 = after the first layer), one forward pass
            layers = [int(x) for x in layers]
            if min(layers) < 1 or (num_layers and max(layers) > int(num_layers)):
                raise ValueError(f"layers {layers} must lie in 1..num_layers")
            self.params["layers"] = layers
        self.batch_tokens = batch_tokens

    def assets(self):
        return [pretrained_dir(self.params["model"])]

    def _hidden_size(self):
        cfg = pretrained_dir(self.params["model"]) / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"{cfg} missing: run  python scripts/get_pretrained.py {self.params['model']}")
        return json.loads(cfg.read_text())["hidden_size"]

    def feature_names(self):
        tag = self.params["model"].split("/")[-1].replace("_UR50D", "")
        names = [f"{tag}_mean_{i}" for i in range(self._hidden_size())]
        if self.params["pooling"] == "mean_max":
            names += [f"{tag}_max_{i}" for i in range(self._hidden_size())]
        return names

    def _load(self):
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from transformers import AutoTokenizer, EsmModel
        d = pretrained_dir(self.params["model"])
        self._hidden_size()  # clear error if weights are missing
        self._tok = AutoTokenizer.from_pretrained(d)
        self._model = EsmModel.from_pretrained(d, add_pooling_layer=False).eval()
        n_keep = self.params.get("num_layers")
        if n_keep:
            if n_keep > len(self._model.encoder.layer):
                raise ValueError(f"num_layers={n_keep} but the model has {len(self._model.encoder.layer)}")
            self._model.encoder.layer = self._model.encoder.layer[:n_keep]
        quant = self.params.get("quantize")
        self._device = "cuda" if torch.cuda.is_available() and not quant else "cpu"  # int8 kernels are CPU-only
        if quant == "int8":
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # deprecation notice for the eager-mode API
                self._model = torch.ao.quantization.quantize_dynamic(self._model, {torch.nn.Linear},
                                                                     dtype=torch.qint8)
        self._model.to(self._device)

    def compute(self, values):
        import torch
        self._load()
        p = self.params
        dim = self._hidden_size()
        n = len(values)
        sums = np.zeros((n, dim), dtype=np.float64)
        maxs = np.full((n, dim), -np.inf, dtype=np.float32)
        counts = np.zeros(n, dtype=np.int64)

        def make_pieces(cap):
            out = []  # (sequence index, window text)
            for i, s in enumerate(values):
                s = s[:cap] if cap else s
                out += [(i, s[st: st + p["window"]]) for st in range(0, max(len(s), 1), p["window"])]
            out.sort(key=lambda t: -len(t[1]))  # similar lengths per batch = little padding
            return out

        # Inference time guard (set by the submission's predict.py; off in training).
        # After a few seconds, extrapolate the total time. If it would exceed the
        # budget, truncate sequences to the length that fits and start over. A
        # slightly worse embedding for long enzymes beats a timed-out submission.
        budget = float(os.environ.get("KCATKM_ESM_BUDGET", 0) or 0)
        left = time_left(reserve=25)  # ESM runs last: keep time only for the model and writing output
        if left is not None:
            if left < 10:
                raise TimeoutError(f"only {left:.0f}s left before the deadline; skipping ESM")
            budget = min(budget, left) if budget else left
        lengths = np.array([len(v) for v in values])
        total = int(np.minimum(lengths, p["max_residues"] or 10**9).sum())
        pieces = make_pieces(p["max_residues"])
        t0, done, checked = time.time(), 0, not budget

        b = 0
        while b < len(pieces):
            elapsed = time.time() - t0
            if not checked and (elapsed > 5 or done > 0.1 * total):
                checked = True
                rate = done / max(elapsed, 1e-6)          # residues/s (longest windows first: pessimistic)
                if total / max(rate, 1e-6) > budget:
                    allowed = max((budget - elapsed) * rate, len(values) * 50)
                    lo, hi = 50, int(lengths.max())
                    while lo < hi:                         # largest cap whose residue count fits
                        mid = (lo + hi + 1) // 2
                        lo, hi = (mid, hi) if np.minimum(lengths, mid).sum() <= allowed else (lo, mid - 1)
                    print(f"[esm] projected {total / rate:.0f}s > budget {budget:.0f}s: truncating sequences "
                          f"to {lo} residues", file=sys.stderr, flush=True)
                    pieces, b = make_pieces(lo), 0
                    STATE["truncated"] = True
                    sums[:], counts[:], maxs[:] = 0, 0, -np.inf
                    continue
            longest = len(pieces[b][1]) + 2
            size = max(1, batch_tokens(self.batch_tokens) // longest)
            batch = pieces[b: b + size]
            b += size
            enc = self._tok([t for _, t in batch], return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.inference_mode():
                out = self._model(**enc, output_hidden_states=p["layer"] != -1 or bool(p.get("layers")))
            if p.get("layers"):
                h = sum(out.hidden_states[k] for k in p["layers"]) / len(p["layers"])
            else:
                h = out.last_hidden_state if p["layer"] == -1 else out.hidden_states[p["layer"]]
            h = h.float().cpu().numpy()
            for row, (i, text) in enumerate(batch):
                res = h[row, 1: len(text) + 1]  # drop <cls> at 0 and <eos>/padding after
                sums[i] += res.sum(axis=0)
                counts[i] += len(res)
                done += len(text)
                if p["pooling"] == "mean_max":
                    maxs[i] = np.maximum(maxs[i], res.max(axis=0))

        mean = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
        if p["pooling"] == "mean_max":
            return np.hstack([mean, maxs])
        return mean


@register
class SmilesTransformer(Extractor):
    """Pretrained SMILES language model (RoBERTa-style, e.g. ChemBERTa), masked mean pooling.

        features:
          - chemberta: {model: DeepChem/ChemBERTa-77M-MTR}

    SMILES longer than the model's context are truncated (rare: few exceed ~500 tokens).
    """
    name = "chemberta"
    entity = "smiles"
    chunk_size = 2048

    def __init__(self, model="DeepChem/ChemBERTa-77M-MTR", layer=-1, batch_size=64):
        super().__init__(model=model, layer=layer)
        self.batch_size = batch_size

    def assets(self):
        return [pretrained_dir(self.params["model"])]

    def _config(self):
        cfg = pretrained_dir(self.params["model"]) / "config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"{cfg} missing: run  python scripts/get_pretrained.py {self.params['model']}")
        return json.loads(cfg.read_text())

    def feature_names(self):
        tag = self.params["model"].split("/")[-1]
        return [f"{tag}_{i}" for i in range(self._config()["hidden_size"])]

    def _load(self):
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        d = pretrained_dir(self.params["model"])
        self._config()
        self._tok = AutoTokenizer.from_pretrained(d)
        self._model = AutoModel.from_pretrained(d, add_pooling_layer=False).eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._max_len = min(512, self._config().get("max_position_embeddings", 514) - 2)

    def compute(self, values):
        import torch
        left = time_left(reserve=15)
        if left is not None and left < 5:
            raise TimeoutError(f"only {left:.0f}s left before the deadline; skipping {self.name}")
        self._load()
        layer = self.params["layer"]
        order = np.argsort([len(v) for v in values])
        out = np.zeros((len(values), self._config()["hidden_size"]), dtype=np.float32)
        for s in range(0, len(values), self.batch_size):
            idx = order[s: s + self.batch_size]
            enc = self._tok([values[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                            max_length=self._max_len, return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.inference_mode():
                res = self._model(**enc, output_hidden_states=layer != -1)
            h = res.last_hidden_state if layer == -1 else res.hidden_states[layer]
            attn = enc["attention_mask"].cpu().float()
            mask = attn * (1 - special.float())   # real SMILES tokens only (no <s>, </s>, padding)
            empty = mask.sum(1) == 0
            mask[empty] = attn[empty]              # degenerate input: pool over everything
            pooled = (h.float().cpu() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            out[idx] = pooled.numpy()
        return out
