"""Compatibility helpers for Hugging Face dataset metadata.

LeRobot metadata exported prior to `datasets>=4.40` still refers to the
legacy ``List`` feature type even though the upstream library now calls
it ``Sequence``.  Registering the alias in one place saves every script
from repeating the same monkey-patch and prevents the runtime
``ValueError`` complaining about unknown feature types.
"""

from __future__ import annotations


def ensure_hf_list_feature() -> None:
    """Register the ``List`` feature alias if the runtime lacks it."""

    try:
        from datasets.features import features as hf_features
    except Exception:
        return

    feature_types = getattr(hf_features, "_FEATURE_TYPES", None)
    if isinstance(feature_types, dict):
        feature_types.setdefault("List", hf_features.Sequence)


# Apply the patch immediately so importing this module is sufficient.
ensure_hf_list_feature()
