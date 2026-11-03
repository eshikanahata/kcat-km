"""kcatkm: shared code for the Leak CURBER joint kcat/Km task.

Everything that both training and inference need lives here, so the exact same
feature code runs locally and inside the submission Docker image.
"""

TARGETS = ("log10_kcat", "log10_km")
INPUT_COLUMNS = ("id", "smiles", "sequence")

__version__ = "0.1.0"
