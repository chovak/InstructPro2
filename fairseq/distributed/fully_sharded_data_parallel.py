# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Optional

import torch
import torch.nn as nn
from fairseq.dataclass.configs import DistributedTrainingConfig
from fairseq.distributed import utils as dist_utils

try:
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
    from fairscale.nn.wrap import wrap, enable_wrap
    has_FSDP = True
except ImportError:
    FSDP = torch.nn.Module
    has_FSDP = False


def _auto_wrap_special_layers(module, mixed_precision_config):
    """
    INTERNAL HELPER: Automatically finds and wraps ProGen/BERT layers 
    into their own FSDP units to prevent OOMs (ZeRO-3).
    """
    # Define the target layers to wrap individually
    target_layers = {
        "ProGenBlock", 
        "BertLayer", 
        "TransformerDecoderLayer", 
        "TransformerEncoderLayer"
    }

    # Recursive function
    def apply_wrap(sub_module):
        for name, child in sub_module.named_children():
            # If the child is one of our target heavy blocks, wrap it!
            if type(child).__name__ in target_layers:
                wrapped_child = wrap(
                    child, 
                    **mixed_precision_config 
                )
                setattr(sub_module, name, wrapped_child)
            else:
                # Otherwise, recurse deeper to find blocks inside
                apply_wrap(child)

    # Apply the wrapping logic
    apply_wrap(module)
    return module


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around fairscale's FullyShardedDataParallel (FSDP) with some
    fairseq-specific checkpoint saving/loading logic.
    """

    def __init__(self, module, *args, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        
        # Clean up arguments that Fairscale might not expect
        kwargs.pop("auto_wrap_policy", None)
        
        super().__init__(module, *args, **kwargs)
        self.use_sharded_state = use_sharded_state

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        if self.flatten_parameters:
            return self.module.module
        else:
            return self.module

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if self.use_sharded_state:
            return super().local_state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        else:
            if self.rank == 0:
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )
            else:
                # We must call state_dict() due to use of communication
                # primitives. But we don't use the result.
                super().state_dict()
                return destination or {}

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        if self.use_sharded_state:
            return super().load_local_state_dict(state_dict, strict=strict)
        else:
            state_dict = dist_utils.broadcast_object(
                state_dict, src_rank=0, group=self.process_group
            )
            return super().load_state_dict(state_dict, strict=strict)


@contextlib.contextmanager
def fsdp_enable_wrap(cfg: DistributedTrainingConfig):
    if not has_FSDP:
        raise ImportError(
            "Cannot find FullyShardedDataParallel. "
            "Please install fairscale with: pip install fairscale"
        )

    # --- HARDCODED BF16 SETTINGS ---
    # Forces BF16 mixed precision to bypass config lookup issues
    compute_dtype = torch.bfloat16
    mixed_precision = True 
    # -------------------------------

    group = dist_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        from fairscale.utils.testing import DummyProcessGroup
        group = DummyProcessGroup(rank=0, size=1)
        
    fsdp_config = {
        "process_group": group,
        "reshard_after_forward": not cfg.no_reshard_after_forward,
        "mixed_precision": mixed_precision,
        "fp32_reduce_scatter": cfg.fp32_reduce_scatter,
        "flatten_parameters": True,
        "cpu_offload": cfg.cpu_offload,
        "compute_dtype": compute_dtype,
        "bucket_cap_mb": cfg.bucket_cap_mb,
        "state_dict_device": torch.device("cpu"),
        # Keep buffers (LayerNorm, etc) in FP32 for stability
        "buffer_dtype": torch.float32, 
    }
    
    with enable_wrap(
        wrapper_cls=FullyShardedDataParallel,
        use_sharded_state=cfg.use_sharded_state,
        **fsdp_config,
    ):
        yield


def fsdp_wrap(module, min_num_params: Optional[int] = None, **kwargs):
    """
    Helper to wrap layers/modules in FSDP. This falls back to a no-op if
    fairscale is not available.
    """
    try:
        from fairscale.nn import wrap
        # Clean up kwargs just in case
        kwargs.pop("auto_wrap_policy", None)

        layer_config = {
            "reshard_after_forward": True,
            "flatten_parameters": True,
            "compute_dtype": torch.bfloat16,
            "mixed_precision": True,
            "buffer_dtype": torch.float32
        }

        # 2. Recursively wrap ProGenBlocks/BertLayers FIRST
        _auto_wrap_special_layers(module, layer_config)
        # ------------------------------
        
        if min_num_params is not None:
            num_params = sum(p.numel() for p in module.parameters())
            if num_params >= min_num_params:
                return wrap(module, **kwargs)
            else:
                return module
        else:
            return wrap(module, **kwargs)
            
    except ImportError:
        return module
# # Copyright (c) Facebook, Inc. and its affiliates.
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.


# # =========================================================================
# # SMART ROUTER: SELECT FSDP ENGINE BASED ON ENVIRONMENT VARIABLE
# # =========================================================================
# # Set via bash: export FSDP_STRATEGY="hybrid" OR "full"
# CURRENT_STRATEGY = os.environ.get("FSDP_STRATEGY", "full").lower()

    