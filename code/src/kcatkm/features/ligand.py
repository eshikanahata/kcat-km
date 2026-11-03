"""Substrate (SMILES) features. RDKit-based extractors import RDKit lazily."""
import numpy as np

from .base import Extractor, register

# Character set from the starter baseline (plus a catch-all bucket).
SMILES_CHARS = list("CNOSPFIBrcnospl()[]=#@+-\\/0123456789.%Hl")
SMILES_CHARS = list(dict.fromkeys(SMILES_CHARS))
CHAR_IDX = {c: i for i, c in enumerate(SMILES_CHARS)}


@register
class SmilesChars(Extractor):
    """Log length + character fractions (the starter's ligand features)."""
    name = "smiles_chars"
    entity = "smiles"

    def compute(self, values):
        k = len(SMILES_CHARS)
        out = np.zeros((len(values), k + 2), dtype=np.float32)
        for i, s in enumerate(values):
            out[i, 0] = np.log1p(len(s))
            for ch in s:
                out[i, 1 + CHAR_IDX.get(ch, k)] += 1
            out[i, 1:] /= max(len(s), 1)
        return out

    def feature_names(self):
        return ["smi_log_len"] + [f"smi_ch_{c}" for c in SMILES_CHARS] + ["smi_ch_other"]


def _mol(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smi)


@register
class Morgan(Extractor):
    """Morgan (ECFP-like) fingerprint. params: radius, n_bits, counts."""
    name = "morgan"
    entity = "smiles"

    def __init__(self, radius=2, n_bits=1024, counts=True):
        super().__init__(radius=radius, n_bits=n_bits, counts=counts)

    def compute(self, values):
        from rdkit.Chem import rdFingerprintGenerator
        p = self.params
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=p["radius"], fpSize=p["n_bits"])
        out = np.zeros((len(values), p["n_bits"] + 1), dtype=np.float32)
        for i, s in enumerate(values):
            m = _mol(s)
            if m is None:
                out[i, -1] = 1  # parse-failure flag
                continue
            fp = gen.GetCountFingerprintAsNumPy(m) if p["counts"] else gen.GetFingerprintAsNumPy(m)
            out[i, :-1] = fp
        return out

    def feature_names(self):
        return [f"morgan_{i}" for i in range(self.params["n_bits"])] + ["morgan_parse_fail"]


@register
class RDKitDescriptors(Extractor):
    """All RDKit 2D descriptors (~210). NaN where RDKit fails; impute in the model."""
    name = "rdkit_desc"
    entity = "smiles"

    def _names(self):
        from rdkit.Chem import Descriptors
        return [n for n, _ in Descriptors._descList]

    def compute(self, values):
        from rdkit.Chem import Descriptors
        fns = Descriptors._descList
        out = np.full((len(values), len(fns)), np.nan, dtype=np.float32)
        for i, s in enumerate(values):
            m = _mol(s)
            if m is None:
                continue
            for j, (_, fn) in enumerate(fns):
                try:
                    out[i, j] = fn(m)
                except Exception:
                    pass
        out[~np.isfinite(out)] = np.nan
        return np.clip(out, -1e6, 1e6)

    def feature_names(self):
        return [f"rd_{n}" for n in self._names()]


@register
class MACCS(Extractor):
    name = "maccs"
    entity = "smiles"

    def compute(self, values):
        from rdkit.Chem import MACCSkeys
        out = np.zeros((len(values), 167), dtype=np.float32)
        for i, s in enumerate(values):
            m = _mol(s)
            if m is not None:
                out[i] = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
        return out

    def feature_names(self):
        return [f"maccs_{i}" for i in range(167)]
