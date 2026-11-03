"""How much does canonicalising SMILES change substrate matching? (seconds)

    python scripts/check_canonical.py
"""
import _bootstrap  # noqa: F401
from kcatkm import data
from kcatkm.labelfeat import canonical

train, val = data.load_train(), data.load_val_inputs()
raw_tr, raw_va = set(train["smiles"]), set(val["smiles"])
can_tr = set(canonical(list(raw_tr)))
can_va = canonical(val["smiles"].tolist())
print(f"train: {len(raw_tr)} raw SMILES -> {len(can_tr)} canonical ({len(raw_tr) - len(can_tr)} merged)")
print(f"val rows whose substrate is seen in train: raw {val['smiles'].isin(raw_tr).mean():.1%} "
      f"-> canonical {sum(c in can_tr for c in can_va) / len(val):.1%}")
