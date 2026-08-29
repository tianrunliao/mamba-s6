"""Compatibility wrapper for the canonical Mamba S6 implementation.

The project originally carried two near-identical model files.  Keeping this
module as a re-export preserves old imports while ensuring every strategy uses
the same tested implementation in :mod:`models.mamba_model`.
"""

from .mamba_model import (
    FEATURE_ORDER,
    MambaBlock,
    MambaTimingNet,
    SelectiveSSM,
    build_training_data,
    create_mamba_model,
    differentiable_mdd_penalty,
    resolve_feature_order,
    train_mamba_model,
    train_mamba_net,
    train_mamba_net_enhanced,
)

__all__ = [
    'FEATURE_ORDER',
    'MambaBlock',
    'MambaTimingNet',
    'SelectiveSSM',
    'build_training_data',
    'create_mamba_model',
    'differentiable_mdd_penalty',
    'resolve_feature_order',
    'train_mamba_model',
    'train_mamba_net',
    'train_mamba_net_enhanced',
]
