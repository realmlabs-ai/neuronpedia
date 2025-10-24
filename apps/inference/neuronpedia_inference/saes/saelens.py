from typing import Any

import torch
from sae_lens.saes.sae import SAE
from sae_lens.registry import SAE_CLASS_REGISTRY
# Import all SAE architectures to ensure they're registered
from sae_lens.saes import batchtopk_sae  # noqa: F401
from sae_lens.saes import gated_sae  # noqa: F401
from sae_lens.saes import jumprelu_sae  # noqa: F401
from sae_lens.saes import standard_sae  # noqa: F401
from sae_lens.saes import topk_sae  # noqa: F401
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

from neuronpedia_inference.saes.base import BaseSAE

# Manually register batchtopk architecture since it's not auto-registered
if 'batchtopk' not in SAE_CLASS_REGISTRY:
    SAE_CLASS_REGISTRY['batchtopk'] = (BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig)

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class SaeLensSAE(BaseSAE):
    @staticmethod
    def load(release: str, sae_id: str, device: str, dtype: str) -> tuple[Any, str]:
        # For root-level SAEs on HuggingFace (empty sae_id), just use the repo name as sae_id
        # SAELens will try to load from {release}/{sae_id}/, so we pass release as sae_id
        if sae_id == "" or sae_id is None:
            # For root-level files, download and load manually
            from huggingface_hub import hf_hub_download
            import json

            # Download cfg.json and sae_weights.safetensors from repo root
            cfg_path = hf_hub_download(repo_id=release, filename="cfg.json")
            weights_path = hf_hub_download(repo_id=release, filename="sae_weights.safetensors")

            # Load config
            with open(cfg_path, 'r') as f:
                cfg_dict = json.load(f)

            # Load SAE from the downloaded files
            import os
            repo_dir = os.path.dirname(cfg_path)
            loaded_sae = SAE.load_from_pretrained(repo_dir, device=device)
        else:
            loaded_sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device,
            )
        loaded_sae.to(device, dtype=DTYPE_MAP[dtype])
        loaded_sae.fold_W_dec_norm()
        loaded_sae.eval()

        return loaded_sae, loaded_sae.cfg.metadata.hook_name
