#!/bin/bash

# Start the inference server with custom Llama 3.1 8B SAE from HuggingFace
# This loads the SAE from realmlabs-ai/llama-3.1-8B-Chat-SAE

cd "$(dirname "$0")"

# Custom SAE configuration in JSON format
# Each config needs:
# - model: the model ID (must match what you'll use in the webapp)
# - set: the set name (this becomes part of the source ID like "20-autointerp-sae")
# - type: "custom-hf" to indicate this is a custom HuggingFace SAE
# - hf_repo_id: the HuggingFace repo ID
# - hook_name: the hook point in the model where this SAE operates
# - saes: list of neuronpedia IDs (these should match your Source IDs in the database)

CUSTOM_SAE_CONFIG='[{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "set": "autointerp-sae",
  "type": "custom-hf",
  "hf_repo_id": "realmlabs-ai/llama-3.1-8B-Chat-SAE",
  "sae_id": "",
  "saes": ["20-autointerp-sae"]
}]'

# Start the server
poetry run python start.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --sae_sets "" \
  --custom_sae_configs "$CUSTOM_SAE_CONFIG" \
  --model_dtype float16 \
  --sae_dtype float16 \
  --device cuda \
  --max_loaded_saes 50
