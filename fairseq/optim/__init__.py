# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from fairseq import registry
from fairseq.optim.fairseq_optimizer import (  # noqa
    FairseqOptimizer,
    LegacyFairseqOptimizer,
)
from omegaconf import DictConfig

__all__ = [
    "AMPOptimizer",
    "FairseqOptimizer",
    "FP16Optimizer",
    "MemoryEfficientFP16Optimizer",
    "shard_",
]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY,
    OPTIMIZER_DATACLASS_REGISTRY,
) = registry.setup_registry("--optimizer", base_class=FairseqOptimizer, required=True)


# def build_optimizer(cfg: DictConfig, params, *extra_args, **extra_kwargs):
#     if all(isinstance(p, dict) for p in params):
#         params = [t for p in params for t in p.values()]
#     params = list(filter(lambda p: p.requires_grad, params))
#     return _build_optimizer(cfg, params, *extra_args, **extra_kwargs)
def build_optimizer(cfg: DictConfig, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        if all("params" in p for p in params):
            # Standard torch-style param groups (e.g. our per-group LR
            # multiplier groups from fp16_optimizer.py's
            # _group_params_by_lr_multiplier). Preserve group structure —
            # just filter each group's own tensor list by requires_grad,
            # and drop any group left empty after filtering.
            new_params = []
            for group in params:
                filtered = list(filter(lambda p: p.requires_grad, group["params"]))
                if filtered:
                    new_group = dict(group)
                    new_group["params"] = filtered
                    new_params.append(new_group)
            params = new_params
        else:
            # Legacy path: dicts mapping arbitrary keys (e.g. device index)
            # to individual flattened parameter tensors.
            flat = [t for p in params for t in p.values()]
            params = list(filter(lambda p: p.requires_grad, flat))
    else:
        params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(cfg, params, *extra_args, **extra_kwargs)


# automatically import any Python files in the optim/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.optim." + file_name)



# Manually register the optimizers so Trainer can find them
try:
    from fairseq.optim.amp_optimizer import AMPOptimizer
    globals()['AMPOptimizer'] = AMPOptimizer
except ImportError:
    pass

try:
    from fairseq.optim.fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
    globals()['FP16Optimizer'] = FP16Optimizer
    globals()['MemoryEfficientFP16Optimizer'] = MemoryEfficientFP16Optimizer
except ImportError:
    pass



