"""Neural models. Registered as joint models: fit(X, Y) / predict(X) -> (n, 2).

    model:
      name: mlp
      params: {hidden: [512, 256], dropout: 0.3, epochs: 40, lr: 0.001, loss: huber}

Design notes
  * Multi-task: one shared trunk, one linear head with 2 outputs (log kcat, log Km),
    so the network can exploit the correlation between the two targets.
  * Inputs: median-imputed, standardised, clipped to +-5 (heavy-tailed counts and
    RDKit descriptors otherwise dominate). Targets are standardised and mapped back.
  * No early stopping: an inner random split would leak near-identical enzymes and
    stop at the wrong time. Train a fixed number of epochs with a cosine schedule
    and tune `epochs` with CV instead.
  * Pickled as plain numpy arrays, so the saved pipeline does not depend on the
    torch version or device it was trained with.
"""
import numpy as np

from .models import register_joint


def _build(d_in, hidden, dropout, d_out=2):
    import torch.nn as nn
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


@register_joint("mlp")
class MultiTaskMLP:
    def __init__(self, hidden=(512, 256), dropout=0.3, input_dropout=0.1, epochs=40, batch_size=256,
                 lr=1e-3, weight_decay=1e-2, loss="huber", huber_delta=1.0, task_weights=(1.0, 1.0),
                 threads=None, seed=42, verbose=False):
        self.hidden, self.dropout, self.input_dropout = list(hidden), dropout, input_dropout
        self.epochs, self.batch_size, self.lr, self.weight_decay = epochs, batch_size, lr, weight_decay
        self.loss, self.huber_delta, self.task_weights = loss, huber_delta, list(task_weights)
        self.threads, self.seed, self.verbose = threads, seed, verbose
        self.state = None

    def _net(self, d_in):
        import torch.nn as nn
        return nn.Sequential(nn.Dropout(self.input_dropout), _build(d_in, self.hidden, self.dropout))

    # --- preprocessing --------------------------------------------------------
    def _prep(self, X):
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        X = np.where(np.isnan(X), self.x_med, X)
        return np.clip((X - self.x_mu) / self.x_sd, -5, 5).astype(np.float32)

    def fit(self, X, Y, sample_weight=None):
        import torch
        import torch.nn as nn
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
        self.x_med = np.nan_to_num(np.nanmedian(X, axis=0)).astype(np.float32)
        Xf = np.where(np.isnan(X), self.x_med, X)
        self.x_mu = Xf.mean(0).astype(np.float32)
        self.x_sd = Xf.std(0).astype(np.float32)
        self.x_sd[self.x_sd < 1e-6] = 1.0
        Xs = self._prep(X)
        self.y_mu, self.y_sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
        Ys = ((Y - self.y_mu) / self.y_sd).astype(np.float32)
        Ms = np.isfinite(Ys).astype(np.float32)  # missing targets (kcat-only extra rows) carry no loss
        Ys = np.nan_to_num(Ys)

        self.d_in = Xs.shape[1]
        net = self._net(self.d_in)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = max(1, self.epochs * int(np.ceil(len(Xs) / self.batch_size)))
        warm = max(1, steps // 20)

        def lr_factor(i):  # linear warm-up for 5% of steps, then cosine decay to 0
            if i < warm:
                return (i + 1) / warm
            return 0.5 * (1 + np.cos(np.pi * min(1.0, (i - warm) / max(1, steps - warm))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        crit = (nn.HuberLoss(delta=self.huber_delta / float(np.mean(self.y_sd)), reduction="none")
                if self.loss == "huber" else nn.MSELoss(reduction="none"))
        w = torch.tensor(self.task_weights, dtype=torch.float32)
        Xt, Yt, Mt = torch.from_numpy(Xs), torch.from_numpy(Ys), torch.from_numpy(Ms)
        sw = None if sample_weight is None else torch.from_numpy(
            (np.asarray(sample_weight, float) / np.mean(sample_weight)).astype(np.float32))

        net.train()
        for ep in range(self.epochs):
            perm = torch.from_numpy(rng.permutation(len(Xs)))
            total = 0.0
            for s in range(0, len(perm), self.batch_size):
                b = perm[s: s + self.batch_size]
                mb = Mt[b]
                per = crit(net(Xt[b]), Yt[b]) * w * mb
                if sw is None:
                    loss = per.sum() / mb.sum().clamp(min=1.0)
                else:
                    loss = (per.sum(1) * sw[b]).sum() / ((mb.sum(1) * sw[b]).sum()).clamp(min=1e-8)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item() * len(b)
            if self.verbose:
                print(f"    epoch {ep + 1}/{self.epochs} loss {total / len(Xs):.4f}", flush=True)

        net.eval()
        self.state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        self.d_in = Xs.shape[1]
        return self

    def predict(self, X):
        import torch
        import torch.nn as nn
        if self.threads:
            torch.set_num_threads(self.threads)
        net = self._net(self.d_in)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self.state.items()})
        net.eval()
        Xs = torch.from_numpy(self._prep(X))
        out = []
        with torch.inference_mode():
            for s in range(0, len(Xs), 4096):
                out.append(net(Xs[s: s + 4096]).numpy())
        return np.vstack(out) * self.y_sd + self.y_mu


PROTEIN_PREFIXES = ("esm2_", "seq_", "aac_", "dpc_")
MOLECULE_PREFIXES = ("ChemBERTa", "morgan_", "rd_", "smi_", "maccs_", "MoLFormer")


def _fusion_net(idx_p, idx_m, idx_o, width, other_width, hidden, dropout, input_dropout):
    import torch
    import torch.nn as nn

    def tower(d_in, d_out):
        return nn.Sequential(nn.Dropout(input_dropout), nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(),
                             nn.Dropout(dropout))

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            for name, idx in (("ip", idx_p), ("im", idx_m), ("io", idx_o)):
                self.register_buffer(name, torch.as_tensor(idx, dtype=torch.long))
            self.protein, self.molecule = tower(len(idx_p), width), tower(len(idx_m), width)
            self.other = tower(len(idx_o), other_width) if len(idx_o) else None
            d = 3 * width + (other_width if len(idx_o) else 0)
            self.head = _build(d, hidden, dropout)

        def forward(self, x):
            p, m = self.protein(x[:, self.ip]), self.molecule(x[:, self.im])
            parts = [p, m, p * m]  # p * m: enzyme x substrate interactions
            if self.other is not None:
                parts.append(self.other(x[:, self.io]))
            return self.head(torch.cat(parts, dim=1))

    return Fusion()


@register_joint("fusion")
class FusionMLP(MultiTaskMLP):
    """Two-tower network: the protein columns and the molecule columns each get their own
    small network; the head sees both, their element-wise product (enzyme x substrate
    interactions) and a third tower over everything else (label-derived features, ...).

        model:
          name: fusion
          params: {width: 128, other_width: 64, hidden: [256], dropout: 0.2, epochs: 40}

    Columns are assigned by feature-name prefix (PROTEIN_PREFIXES / MOLECULE_PREFIXES); the
    pipeline passes the names in via set_feature_names before fitting."""

    def __init__(self, width=128, other_width=64, hidden=(256,), protein_prefixes=PROTEIN_PREFIXES,
                 molecule_prefixes=MOLECULE_PREFIXES, **kw):
        kw.setdefault("dropout", 0.2)
        super().__init__(hidden=hidden, **kw)
        self.width, self.other_width = int(width), int(other_width)
        self.protein_prefixes, self.molecule_prefixes = tuple(protein_prefixes), tuple(molecule_prefixes)
        self.names = None

    def set_feature_names(self, names):
        self.names = list(names)

    def _groups(self, d_in):
        if self.names is None or len(self.names) != d_in:
            raise ValueError("fusion needs the feature names (set by the pipeline) matching X's columns")
        p = [i for i, n in enumerate(self.names) if n.startswith(self.protein_prefixes)]
        m = [i for i, n in enumerate(self.names) if n.startswith(self.molecule_prefixes)]
        o = [i for i, n in enumerate(self.names) if i not in set(p) | set(m)]
        if not p or not m:
            raise ValueError(f"fusion found {len(p)} protein and {len(m)} molecule columns; check the prefixes")
        return p, m, o

    def _net(self, d_in):
        p, m, o = self._groups(d_in)
        return _fusion_net(p, m, o, self.width, self.other_width, self.hidden, self.dropout, self.input_dropout)
