# SAE Integration Guide for Neuronpedia

This guide explains how to integrate your own trained SAEs with the Neuronpedia codebase for generating explanations.

## Overview

The Neuronpedia autointerp system can generate explanations for your custom SAEs. It uses the `sae-auto-interp` library (forked version at `https://github.com/hijohnnylin/sae-auto-interp`) to process SAE features and generate natural language explanations.

## Your SAE Setup Compatibility

**Your Configuration:** Llama 3.1 8B with 4096 hidden states → 32k SAE features (8x overcomplete)
- ✅ **Fully Compatible** - This system is designed for exactly this type of setup

## Key Components

### 1. DefaultExplainer Source
- **Import:** `from sae_auto_interp.explainers import DefaultExplainer`
- **Location:** External dependency defined in `/Users/akash/Desktop/code/neuronpedia/apps/autointerp/pyproject.toml:18`
- **Usage:** `/Users/akash/Desktop/code/neuronpedia/apps/autointerp/neuronpedia_autointerp/routes/explain/default.py:31`

### 2. Data Format Requirements

#### NPActivation Structure
```python
NPActivation(
    tokens=["The", "cat", "sat", "on", "the", "mat"],     # List of token strings
    values=[0.1,   0.8,   0.0,   2.3,   0.1,   0.5]      # Activation values per token
)
```

**What this represents:**
- One specific feature's activations across a sequence of tokens
- Each `NPActivation` = one text example for one feature
- Multiple `NPActivation` objects needed per feature for explanation generation

#### Example for Feature Explanation
```python
# Feature 1234's activations across different texts
activations = [
    NPActivation(tokens=["The", "cat", "meowed"], values=[0.1, 2.1, 0.3]),
    NPActivation(tokens=["A", "dog", "barked"], values=[0.0, 1.8, 0.2]),
    NPActivation(tokens=["The", "bird", "chirped"], values=[0.1, 2.4, 0.1])
]
```

### 3. Dataset for SAE Activations

**Common Datasets Used:**
- `"monology/pile-uncopyrighted"` (most common in examples)
- Any HuggingFace dataset via `--prompts-huggingface-dataset-path`

**Key Point:** You can use any HuggingFace dataset - the system will generate activations from your chosen text corpus.

## What You Need

### Required Components
1. **Your trained SAE model** ✅ (you have this)
2. **HuggingFace dataset** for text input
   - Use `"monology/pile-uncopyrighted"` (standard)
   - Or any dataset relevant to your use case
   - Or the same dataset your SAE was trained on
3. **API access** to LLM service (OpenRouter, OpenAI) for explanation generation

### Dependencies
- `sae-auto-interp` library (automatically installed via poetry)
- FastAPI server for autointerp endpoints
- Compatible with TransformerLens model loading

## Usage Options

### Option 1: Automated Pipeline (Recommended)
Use the built-in dashboard generation scripts:

```bash
cd utils/neuronpedia-utils
python neuronpedia_utils/generate-dashboards-as-saelens.py \
    --model-name="meta-llama/Meta-Llama-3.1-8B" \
    --sae-path="your_sae_path" \
    --np-set-name="your-sae-set-name" \
    --dataset-path="monology/pile-uncopyrighted" \
    --n-prompts=8192 \
    --n-tokens-in-prompt=128
```

**What this does:**
1. Loads your SAE model
2. Runs text through model + SAE to get activations
3. For each feature (0 to 32k):
   - Finds top-K text examples where feature activated strongly
   - Extracts token-level activations
   - Formats as `NPActivation` objects
4. Packages everything for explanation generation

### Option 2: Direct API Usage
Start the autointerp server and make API calls:

```bash
cd apps/autointerp
poetry install . && poetry run python server.py
```

Then POST to `/explain/default` with:
```json
{
    "activations": [/* NPActivation objects */],
    "openrouter_key": "your_api_key",
    "model": "anthropic/claude-3-haiku"
}
```

## Library Automation

**The `sae-auto-interp` library handles:**
- SAE model loading and inference
- Activation extraction across text corpus
- Finding top activating examples per feature
- Token-level activation value extraction
- Data formatting for explanation generation

**You only need to:**
- Provide your SAE model path/config
- Choose your text dataset
- Specify output parameters

## File Locations

**Key Files:**
- Main explanation endpoint: `/Users/akash/Desktop/code/neuronpedia/apps/autointerp/neuronpedia_autointerp/routes/explain/default.py`
- Dependencies: `/Users/akash/Desktop/code/neuronpedia/apps/autointerp/pyproject.toml:18`
- Dashboard generation: `/Users/akash/Desktop/code/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/generate-dashboards-as-saelens.py`
- Data models: `/Users/akash/Desktop/code/neuronpedia/packages/python/neuronpedia-autointerp-client/neuronpedia_autointerp_client/models/`

## Next Steps

1. **Test with existing SAE:** Try the automated pipeline with a known working SAE first
2. **Adapt for your SAE:** Modify the SAE loading code if your format differs from standard
3. **Generate explanations:** Use the explanation API with your processed features
4. **Integrate with webapp:** Import generated data into the Neuronpedia interface

The system is designed to be SAE-agnostic, so your Llama 3.1 8B SAE should work with minimal modifications to the loading pipeline.