"""Hand-crafted enzyme-sequence features."""
from itertools import product

import numpy as np

from .base import Extractor, register

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy and residue masses (Da) for simple global descriptors.
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
MASS = dict(A=71.08, R=156.19, N=114.10, D=115.09, C=103.14, Q=128.13, E=129.12, G=57.05,
            H=137.14, I=113.16, L=113.16, K=128.17, M=131.19, F=147.18, P=97.12, S=87.08,
            T=101.10, W=186.21, Y=163.18, V=99.13)


@register
class SeqLength(Extractor):
    name = "seq_len"
    entity = "sequence"

    def compute(self, values):
        n = np.array([len(s) for s in values], dtype=np.float32)
        return np.stack([n, np.log1p(n)], axis=1)

    def feature_names(self):
        return ["seq_len", "seq_log_len"]


@register
class AAComposition(Extractor):
    """Fraction of each of the 20 amino acids, plus fraction of anything else."""
    name = "aac"
    entity = "sequence"

    def compute(self, values):
        out = np.zeros((len(values), 21), dtype=np.float32)
        for i, s in enumerate(values):
            for ch in s:
                out[i, AA_IDX.get(ch, 20)] += 1
            out[i] /= max(len(s), 1)
        return out

    def feature_names(self):
        return [f"aac_{a}" for a in AA] + ["aac_other"]


@register
class DipeptideComposition(Extractor):
    """Fraction of each of the 400 ordered amino-acid pairs."""
    name = "dpc"
    entity = "sequence"

    def compute(self, values):
        out = np.zeros((len(values), 400), dtype=np.float32)
        for i, s in enumerate(values):
            idx = [AA_IDX.get(c, -1) for c in s]
            for a, b in zip(idx, idx[1:]):
                if a >= 0 and b >= 0:
                    out[i, a * 20 + b] += 1
            out[i] /= max(len(s) - 1, 1)
        return out

    def feature_names(self):
        return [f"dpc_{a}{b}" for a, b in product(AA, AA)]


@register
class SeqPhysChem(Extractor):
    """A few global physico-chemical descriptors."""
    name = "seq_physchem"
    entity = "sequence"

    def compute(self, values):
        rows = []
        for s in values:
            n = max(len(s), 1)
            kd = [KD[c] for c in s if c in KD]
            pos = sum(s.count(c) for c in "KR") + 0.1 * s.count("H")
            neg = sum(s.count(c) for c in "DE")
            arom = sum(s.count(c) for c in "FWY")
            mass = sum(MASS.get(c, 110.0) for c in s) + 18.02
            rows.append([np.mean(kd) if kd else 0.0, (pos - neg) / n, arom / n,
                         np.log10(mass), s.count("C") / n])
        return np.asarray(rows, dtype=np.float32)

    def feature_names(self):
        return ["seq_gravy", "seq_net_charge_frac", "seq_aromatic_frac", "seq_log_mass", "seq_cys_frac"]
