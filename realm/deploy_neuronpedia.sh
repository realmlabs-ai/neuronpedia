#!/bin/bash

# Neuronpedia Local Deployment Script
# 
# Automates the complete setup and deployment of a local Neuronpedia instance,
# including database initialization, webapp building, and SAE data population.
#
# Usage: ./deploy_neuronpedia_local.sh <sae_data_directory>
# Example: ./deploy_neuronpedia_local.sh /home/ubuntu/20-realm-l20r-8x

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBAPP_DIR="${WEBAPP_DIR:-/home/ubuntu/neuronpedia/apps/webapp}"
SOURCE_SET_NAME="${SOURCE_SET_NAME:-autointerp-sae}"

# Helper functions
log_info() {
    echo -e "${BLUE}🔄 $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if PostgreSQL is running
    if ! pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; then
        log_error "PostgreSQL is not running or not accessible"
        log_error "Please start PostgreSQL: sudo systemctl start postgresql"
        exit 1
    fi
    
    # Test database connection
    if ! psql -h localhost -U postgres -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL database"
        log_error "Please check your credentials (postgres/postgres)"
        exit 1
    fi
    
    # Check if webapp directory exists
    if [ ! -d "$WEBAPP_DIR" ]; then
        log_error "Webapp directory not found: $WEBAPP_DIR"
        log_error "Please set WEBAPP_DIR environment variable or update the script"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

setup_database() {
    log_info "Setting up database schema..."
    
    cd "$WEBAPP_DIR"
    
    # Install webapp dependencies
    log_info "Installing webapp dependencies..."
    npm install
    
    # Run database migrations
    log_info "Running Prisma database migrations..."
    npm run migrate:localhost
    
    log_success "Database schema setup complete"
}

build_webapp() {
    log_info "Building webapp..."
    
    cd "$WEBAPP_DIR"
    
    # Build webapp for localhost
    log_info "Building webapp for localhost..."
    npm run build:localhost
    
    log_success "Webapp build complete"
}

upload_data() {
    local sae_data_dir="$1"
    
    log_info "Uploading SAE data..."
    
    # Upload explanations (creates base structure)
    if [ -d "$sae_data_dir/explanations" ]; then
        log_info "Uploading explanations..."
        python "$SCRIPT_DIR/upload_explanations.py" "$sae_data_dir/explanations" "$SOURCE_SET_NAME"
        log_success "Explanations uploaded"
    else
        log_warning "No explanations directory found, skipping..."
    fi
    
    # Upload activations (optional)
    if [ -d "$sae_data_dir/activations" ]; then
        log_info "Uploading activations (this may take 20-30 minutes)..."
        python "$SCRIPT_DIR/upload_activations.py" "$sae_data_dir/activations" "$SOURCE_SET_NAME" --skip-pruning
        log_success "Activations uploaded"
    else
        log_warning "No activations directory found, skipping..."
    fi
    
    # Upload feature statistics (optional)
    if [ -d "$sae_data_dir/features" ]; then
        log_info "Uploading feature statistics (this may take 10-15 minutes)..."
        python "$SCRIPT_DIR/upload_feature_stats.py" "$sae_data_dir/features" "$SOURCE_SET_NAME"
        log_success "Feature statistics uploaded"
    else
        log_warning "No features directory found, skipping..."
    fi
    
    # Fix maxActApprox values
    log_info "Fixing maxActApprox values..."
    python "$SCRIPT_DIR/fix_max_act.py"
    log_success "MaxActApprox values fixed"
}

get_model_info() {
    local sae_data_dir="$1"
    
    # Try to extract model ID from data
    for data_type in explanations features activations; do
        if [ -d "$sae_data_dir/$data_type" ]; then
            local first_file=$(find "$sae_data_dir/$data_type" -name "batch-*.jsonl*" | head -1)
            if [ -n "$first_file" ]; then
                if [[ "$first_file" == *.gz ]]; then
                    MODEL_ID=$(zcat "$first_file" | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['modelId'])" 2>/dev/null || echo "")
                else
                    MODEL_ID=$(head -1 "$first_file" | python3 -c "import sys, json; print(json.load(sys.stdin)['modelId'])" 2>/dev/null || echo "")
                fi
                if [ -n "$MODEL_ID" ]; then
                    # Clean model ID for URL
                    CLEAN_MODEL_ID=$(echo "$MODEL_ID" | tr '/' '-' | tr '_' '-' | tr '[:upper:]' '[:lower:]')
                    break
                fi
            fi
        fi
    done
}

main() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <sae_data_directory>"
        echo "Example: $0 /home/ubuntu/20-realm-l20r-8x"
        echo ""
        echo "Prerequisites:"
        echo "- PostgreSQL running on localhost:5432 with postgres/postgres"
        echo "- Node.js and npm installed"
        echo "- Python with required dependencies (psycopg2)"
        exit 1
    fi
    
    local sae_data_dir="$1"
    
    if [ ! -d "$sae_data_dir" ]; then
        log_error "SAE data directory not found: $sae_data_dir"
        exit 1
    fi
    
    echo "🚀 Starting Neuronpedia Local Deployment"
    echo "=========================================="
    echo "SAE data directory: $sae_data_dir"
    echo "Webapp directory: $WEBAPP_DIR"
    echo "Source set name: $SOURCE_SET_NAME"
    echo ""
    
    # Step 1: Check prerequisites
    check_prerequisites
    
    # Step 2: Set up database
    setup_database
    
    # Step 3: Build webapp
    build_webapp
    
    # Step 4: Upload data
    upload_data "$sae_data_dir"
    
    # Step 5: Get model info for final URLs
    get_model_info "$sae_data_dir"
    
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================="
    
    if [ -n "$CLEAN_MODEL_ID" ]; then
        echo "📊 View your features at:"
        echo "   http://localhost:3000/$CLEAN_MODEL_ID"
        echo "   http://localhost:3000/$CLEAN_MODEL_ID/20-$SOURCE_SET_NAME/311"
    else
        echo "📊 View at: http://localhost:3000"
    fi
    
    echo ""
    log_info "Starting webapp..."
    echo "📋 The webapp will start on http://localhost:3000"
    echo "📋 Use Ctrl+C to stop the server"
    echo ""
    
    # Start webapp (this will run indefinitely)
    cd "$WEBAPP_DIR"
    npm run dev:localhost
}

# Run main function with all arguments
main "$@"