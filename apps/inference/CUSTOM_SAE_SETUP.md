# Custom SAE Setup Guide

This guide explains how to set up Neuronpedia to use custom SAEs from HuggingFace, particularly for SAEs with architectures not in the standard SAELens registry (like batchtopk).

## Overview

The Neuronpedia inference server can load custom SAEs directly from HuggingFace repositories. This is useful when:
- You've trained your own SAE
- Your SAE uses a newer architecture (like batchtopk)
- Your SAE isn't in the official SAELens registry

## What Was Modified

### 1. Inference Server Code

**File: `neuronpedia_inference/saes/saelens.py`**
- Added imports for all SAE architecture modules (batchtopk, gated, jumprelu, etc.)
- Manually registered batchtopk architecture in the SAE_CLASS_REGISTRY
- Added logic to handle root-level SAEs on HuggingFace (files at repo root vs. in subdirectories)

**File: `neuronpedia_inference/config.py`**
- Added `custom_sae_configs` parameter to accept JSON configs for custom HF SAEs
- Modified `_generate_sae_config()` to merge custom configs with SAELens registry configs

**File: `neuronpedia_inference/sae_manager.py`**
- Updated `load_sae()` to detect and load custom HF SAEs differently from registry SAEs

**File: `neuronpedia_inference/server.py`**
- Added `custom_sae_configs` parameter to Config initialization

**Files: `start.py`, `neuronpedia_inference/args.py`**
- Added `--custom_sae_configs` command-line argument and environment variable support

### 2. Python Dependencies

**File: `pyproject.toml`**
- Updated SAELens from 6.12.1 to 6.18.0 for batchtopk support

### 3. Database Configuration

**Model Table:**
- Set `inferenceEnabled = true` for the model
- Set `tlensId` field to the HuggingFace model ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

**Source Table:**
- Set `inferenceEnabled = true` for all SAE sources

### 4. Webapp Configuration

**File: `apps/webapp/.env.localhost`**
- Added `USE_LOCALHOST_INFERENCE=true`
- Added `INFERENCE_SERVER_SECRET=local-dev-secret`

## Custom SAE Configuration Format

The custom SAE config is a JSON array with the following structure:

```json
[{
  "model": "your-database-model-id",
  "set": "your-source-set-name",
  "type": "custom-hf",
  "hf_repo_id": "your-org/your-sae-repo",
  "sae_id": "",
  "saes": ["layer-source-id"]
}]
```

**Fields:**
- `model`: The model ID as it appears in your database (e.g., `meta-llama-llama-3.1-8b-instruct`)
- `set`: The source set name (e.g., `autointerp-sae`)
- `type`: Must be `"custom-hf"` for custom HuggingFace SAEs
- `hf_repo_id`: HuggingFace repo ID where the SAE is stored
- `sae_id`: Path within the repo to SAE files (use `""` for root-level files)
- `saes`: Array of neuronpedia source IDs (e.g., `["20-autointerp-sae"]`)

## Step-by-Step Setup

### 1. Prepare Your SAE on HuggingFace

Your HuggingFace repo should contain:
- `cfg.json` - SAE configuration file
- `sae_weights.safetensors` - SAE weights

These can be at the root of the repo or in a subdirectory.

### 2. Update Dependencies

```bash
cd apps/inference
# Update SAELens to 6.18.0+
poetry update sae-lens
poetry install
```

### 3. Configure Database

```bash
# Enable inference for your model
psql "postgres://postgres:postgres@localhost:5432/postgres" -c \
  "UPDATE \"Model\" SET \"inferenceEnabled\" = true, \"tlensId\" = 'meta-llama/Llama-3.1-8B-Instruct' WHERE id = 'meta-llama-llama-3.1-8b-instruct';"

# Enable inference for all SAE sources
psql "postgres://postgres:postgres@localhost:5432/postgres" -c \
  "UPDATE \"Source\" SET \"inferenceEnabled\" = true WHERE \"modelId\" = 'meta-llama-llama-3.1-8b-instruct';"
```

**Important:** Set the `tlensId` to match the HuggingFace model ID format that TransformerLens expects.

### 4. Configure Webapp

Add to `apps/webapp/.env.localhost`:

```bash
USE_LOCALHOST_INFERENCE=true
INFERENCE_SERVER_SECRET=local-dev-secret
```

### 5. Create Startup Script

Create a script like `start_llama_sae.sh`:

```bash
#!/bin/bash

cd "$(dirname "$0")"

CUSTOM_SAE_CONFIG='[{
  "model": "meta-llama-llama-3.1-8b-instruct",
  "set": "autointerp-sae",
  "type": "custom-hf",
  "hf_repo_id": "realmlabs-ai/llama-3.1-8B-Chat-SAE",
  "sae_id": "",
  "saes": ["20-autointerp-sae"]
}]'

poetry run python start.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --sae_sets "" \
  --custom_sae_configs "$CUSTOM_SAE_CONFIG" \
  --model_dtype float16 \
  --sae_dtype float16 \
  --device cuda \
  --max_loaded_saes 50
```

### 6. Start Services

```bash
# Start inference server
cd apps/inference
./start_llama_sae.sh

# Start webapp (in another terminal)
cd apps/webapp
npm run dev:localhost
```

## Testing

1. Navigate to a feature page: `http://localhost:3000/meta-llama-llama-3.1-8b-instruct/20-autointerp-sae/0`
2. You should see the activation test form
3. Enter test text and click "Test"
4. The webapp will call the local inference server which uses your custom SAE

## Troubleshooting

### "Unsupported model" Error

**Problem:** Inference server returns 400 with "Unsupported model"

**Solution:** Ensure the `tlensId` in the Model table matches what you're passing as `--model_id` to the inference server, OR matches the `--custom_hf_model_id` parameter.

### "KeyError: 'batchtopk'" Error

**Problem:** SAE architecture not recognized

**Solutions:**
1. Ensure SAELens is updated to 6.18.0+: `poetry update sae-lens`
2. Check that the manual registration is in `saelens.py` (already done in this setup)
3. Verify the imports are working:
   ```bash
   python -c "import neuronpedia_inference.saes.saelens; from sae_lens.registry import SAE_CLASS_REGISTRY; print(list(SAE_CLASS_REGISTRY.keys()))"
   ```
   Should include `'batchtopk'`

### SAE Files Not Found (404)

**Problem:** HuggingFace returns 404 when loading SAE files

**Solutions:**
1. Check `hf_repo_id` is correct
2. Verify `sae_id` path:
   - Use `""` for root-level files
   - Use folder name (e.g., `"blocks.20.hook_resid_post"`) if files are in a subdirectory
3. Ensure the repo is public or you're authenticated

### Model ID Mismatch

**Problem:** Database model ID doesn't match what inference server expects

**Solution:** Use the `tlensId` field in the Model table to map database IDs to HuggingFace/TransformerLens IDs.

## Architecture Notes

### Why batchtopk Needs Manual Registration

SAELens 6.18.0 includes the batchtopk architecture module but doesn't auto-register it in the `SAE_CLASS_REGISTRY`. Our code manually registers it:

```python
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

if 'batchtopk' not in SAE_CLASS_REGISTRY:
    SAE_CLASS_REGISTRY['batchtopk'] = (BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig)
```

### Root-Level vs Subdirectory SAEs

SAELens expects SAE files in a specific structure:
- **Subdirectory structure:** `{repo}/{sae_id}/cfg.json` and `{repo}/{sae_id}/sae_weights.safetensors`
- **Root-level structure:** `{repo}/cfg.json` and `{repo}/sae_weights.safetensors`

Our code handles root-level SAEs by:
1. Using `hf_hub_download()` to download files from repo root
2. Getting the cache directory where files are stored
3. Using `SAE.load_from_pretrained(cache_dir)` to load from local cache

## Example: Llama 3.1 8B with Custom SAE

This setup uses:
- **Base Model:** `meta-llama/Llama-3.1-8B-Instruct` (from HuggingFace)
- **Custom SAE:** `realmlabs-ai/llama-3.1-8B-Chat-SAE` (batchtopk architecture)
- **Hook Point:** `blocks.20.hook_resid_post` (layer 20 residual stream)
- **Database IDs:** Model ID `meta-llama-llama-3.1-8b-instruct`, Source ID `20-autointerp-sae`

The mapping works as:
1. Webapp reads from database: `meta-llama-llama-3.1-8b-instruct`
2. Webapp looks up `tlensId`: `meta-llama/Llama-3.1-8B-Instruct`
3. Webapp sends to inference server: `model: "meta-llama/Llama-3.1-8B-Instruct"`
4. Inference server loads base model from HF: `meta-llama/Llama-3.1-8B-Instruct`
5. Inference server loads custom SAE from HF: `realmlabs-ai/llama-3.1-8B-Chat-SAE`
