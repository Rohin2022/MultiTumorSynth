import torch
from pytorch_lightning.utilities.upgrade_checkpoint import _upgrade

# 1. Import all the specific classes PyTorch is complaining about
from omegaconf import DictConfig, ListConfig 
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from typing import Any  # <-- New import
from collections import defaultdict

# 2. Add them ALL to the safe globals list
torch.serialization.add_safe_globals([
    DictConfig, 
    Metadata,
    list,
    int,
    ListConfig, 
    ContainerMetadata, 
    Any,  # <-- New addition
    dict,
    defaultdict,
    AnyNode
])

class Args:
    path = "TumorGeneration/model_weight/AutoencoderModel.ckpt"
    extension = ".ckpt"
    map_to_cpu = True

try:
    print(f"Attempting to upgrade: {Args.path}...")
    _upgrade(Args())
    print("\n✅ Success! The checkpoint is now compatible with v2.6.1.")
except Exception as e:
    print(f"\n❌ Failed: {e}")