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
        import os
        import json

        # Check if this is a local file path
        is_local_path = (
            release.startswith('/') or 
            release.startswith('~') or 
            release.startswith('file://') or
            os.path.exists(os.path.expanduser(release))
        )

        if sae_id == "" or sae_id is None:
            if is_local_path:
                # Handle local file path
                local_path = release.replace('file://', '')
                local_path = os.path.expanduser(local_path)
                
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Local SAE path does not exist: {local_path}")
                
                cfg_path = os.path.join(local_path, "cfg.json")
                if not os.path.exists(cfg_path):
                    raise FileNotFoundError(f"cfg.json not found in: {local_path}")

                # Load config
                with open(cfg_path, 'r') as f:
                    cfg_dict = json.load(f)

                # Check for different weight file formats
                weights_files = [
                    "sae_weights.safetensors",
                    "sae_weights.pth", 
                    "weights.safetensors",
                    "weights.pth"
                ]
                
                # Look for any .pth files if standard names not found
                if not any(os.path.exists(os.path.join(local_path, wf)) for wf in weights_files):
                    pth_files = [f for f in os.listdir(local_path) if f.endswith('.pth')]
                    if pth_files:
                        weights_files.append(pth_files[0])  # Use first .pth file found
                
                weights_path = None
                for weight_file in weights_files:
                    full_path = os.path.join(local_path, weight_file)
                    if os.path.exists(full_path):
                        weights_path = full_path
                        break
                
                if weights_path is None:
                    available_files = os.listdir(local_path)
                    raise FileNotFoundError(
                        f"No SAE weights file found in: {local_path}\n"
                        f"Available files: {available_files}\n"
                        f"Expected one of: {weights_files}"
                    )

                # Load SAE from local path
                try:
                    loaded_sae = SAE.load_from_pretrained(local_path, device=device)
                except Exception as e:
                    # If standard loading fails, try loading with explicit weight file
                    print(f"Standard loading failed: {e}")
                    print(f"Attempting to load weights from: {weights_path}")
                    
                    # Try loading weights manually if needed
                    if weights_path.endswith('.pth'):
                        # Convert .pth to safetensors format temporarily
                        import torch
                        from safetensors.torch import save_file
                        
                        print(f"Converting .pth file to safetensors format...")
                        state_dict = torch.load(weights_path, map_location='cpu')
                        print(f"Loaded .pth file with keys: {list(state_dict.keys())}")
                        
                        # Map parameter names to SAELens format
                        mapped_state_dict = {}
                        for key, value in state_dict.items():
                            if key == "encoder_linear.weight":
                                mapped_state_dict["W_enc"] = value
                            elif key == "encoder_linear.bias":
                                mapped_state_dict["b_enc"] = value
                            elif key == "decoder_linear.weight":
                                mapped_state_dict["W_dec"] = value
                            elif key == "decoder_linear.bias":
                                mapped_state_dict["b_dec"] = value
                            else:
                                # Keep other keys as-is
                                mapped_state_dict[key] = value
                        
                        print(f"Mapped to SAELens format with keys: {list(mapped_state_dict.keys())}")
                        
                        # Save as safetensors in the same directory
                        safetensors_path = os.path.join(local_path, "sae_weights.safetensors")
                        save_file(mapped_state_dict, safetensors_path)
                        print(f"Converted to safetensors: {safetensors_path}")
                        
                        # Now try loading again
                        loaded_sae = SAE.load_from_pretrained(local_path, device=device)
                    else:
                        raise
                        
            else:
                # For HuggingFace repos with root-level files
                from huggingface_hub import hf_hub_download
                
                # Download cfg.json and sae_weights.safetensors from repo root
                cfg_path = hf_hub_download(repo_id=release, filename="cfg.json")
                weights_path = hf_hub_download(repo_id=release, filename="sae_weights.safetensors")

                # Load config
                with open(cfg_path, 'r') as f:
                    cfg_dict = json.load(f)

                # Load SAE from the downloaded files
                repo_dir = os.path.dirname(cfg_path)
                loaded_sae = SAE.load_from_pretrained(repo_dir, device=device)
        else:
            # Standard SAELens loading from repo with subdirectory
            loaded_sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=device,
            )
        
        loaded_sae.to(device, dtype=DTYPE_MAP[dtype])
        loaded_sae.fold_W_dec_norm()
        loaded_sae.eval()

        return loaded_sae, loaded_sae.cfg.metadata.hook_name
