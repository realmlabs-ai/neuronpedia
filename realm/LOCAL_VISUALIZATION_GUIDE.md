# Local SAE Visualization Guide

This guide shows you how to visualize your batch-autointerp explanations locally using the Neuronpedia webapp.

## Prerequisites

1. **PostgreSQL database running locally**
2. **Your batch-autointerp explanations** (in `batch-*.jsonl` files)
3. **Node.js and npm installed**

## Step-by-Step Setup

### 1. Start PostgreSQL Database

Make sure you have PostgreSQL running locally with these default settings:
- Host: `localhost`
- Port: `5432`
- Database: `postgres`
- Username: `postgres`
- Password: `postgres`

```bash
# If using Ubuntu:
sudo systemctl start postgresql

# If using Homebrew on macOS:
brew services start postgresql

# Or using Docker:
docker run --name postgres-local -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres
```

### 2. Start the Local Webapp

```bash
cd apps/webapp
npm install  # if you haven't already
npm run dev:localhost
```

The webapp will be available at http://localhost:3000

### 3. Deploy Complete Setup (Recommended)

Use the automated deployment script:

```bash
# Complete deployment with all data types
python realm/deploy_neuronpedia_local.py /path/to/your/sae_data_directory
```

Example:
```bash
python realm/deploy_neuronpedia_local.py /Users/akash/Desktop/code/20-realm-l20r-8x
```

### 3b. Manual Upload (Alternative)

Or upload data manually:

```bash
# Upload explanations only
python realm/upload_explanations.py /path/to/explanations autointerp-sae

# Upload activations (optional)
python realm/upload_activations.py /path/to/activations autointerp-sae

# Upload feature statistics (optional)
python realm/upload_feature_stats.py /path/to/features autointerp-sae
```

### 4. View Your Features

After upload completes, you can view your features at:

- **Model page**: `http://localhost:3000/{model_id}`
- **Source page**: `http://localhost:3000/{model_id}/{layer}-{source_set_name}`
- **Individual features**: `http://localhost:3000/{model_id}/{layer}-{source_set_name}/{feature_index}`

## What the Deployment Script Does

1. **Checks prerequisites** (PostgreSQL connection)
2. **Sets up database schema** (runs Prisma migrations)
3. **Builds the webapp** (installs dependencies and builds)
4. **Auto-detects your model** from the data files
5. **Creates the model** in your local database (if it doesn't exist)
6. **Uploads explanations** in batches to avoid timeouts
7. **Uploads activations** for rich feature visualization
8. **Updates feature statistics** (pos/neg logits, density, histograms)
9. **Fixes display issues** (maxActApprox values)
10. **Starts the webapp** automatically
11. **Provides direct URLs** to view your features

## Script Features

- ✅ **Auto-detection**: Automatically detects model ID and number of layers
- ✅ **Batch processing**: Handles large numbers of explanations efficiently
- ✅ **Error handling**: Continues on errors and reports what succeeded
- ✅ **Progress tracking**: Shows upload progress for each layer
- ✅ **Direct links**: Provides URLs to immediately view your features

## Troubleshooting

### "Local webapp not detected"
Make sure the webapp is running: `cd apps/webapp && npm run dev:localhost`
Or use the deployment script which starts it automatically.

### Database connection errors
Verify PostgreSQL is running and accessible at `localhost:5432`
- Ubuntu: `sudo systemctl status postgresql`
- macOS: `brew services list | grep postgresql`
- Docker: `docker ps | grep postgres`

### Upload failures
- Check that your data files are valid JSONL (gzipped or not)
- Ensure Python has psycopg2 installed: `pip install psycopg2-binary`
- Check file permissions on your data directory
- Ensure the data contains required fields (`modelId`, `layer`, `index`, `description`)

## File Structure

Your SAE data directory should contain:
```
sae_data_directory/
├── explanations/     # Feature explanations
│   ├── batch-0.jsonl.gz
│   ├── batch-1.jsonl.gz
│   └── ...
├── activations/      # Activation patterns
│   ├── batch-0.jsonl.gz
│   ├── batch-1.jsonl.gz
│   └── ...
└── features/         # Feature statistics
    ├── batch-0.jsonl.gz
    ├── batch-1.jsonl.gz
    └── ...
```

Each JSONL file contains explanations like:
```json
{"modelId": "llama-8b-instruct", "layer": "20", "index": "1234", "description": "responds to dog-related concepts", "typeName": "oai_token-act-pair", "explanationModelName": "gpt-4o-mini"}
```

## Next Steps

Once uploaded, you can:
- Browse features by layer and index
- Search explanations by keywords
- View activation patterns (if you also upload activations)
- Compare features across different layers