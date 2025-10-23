#!/usr/bin/env python3
"""
Neuronpedia Local Deployment Script

Automates the complete setup and deployment of a local Neuronpedia instance,
including database initialization, webapp building, and SAE data population.

This script provides a one-command solution for deploying Neuronpedia locally
with sparse autoencoder (SAE) interpretability data.

Usage:
    python deploy_neuronpedia_local.py <sae_data_directory>

Example:
    python deploy_neuronpedia_local.py /Users/akash/Desktop/code/20-realm-l20r-8x

Prerequisites:
    - PostgreSQL server running on localhost:5432 (postgres/postgres credentials)
    - Node.js and npm installed for webapp building
    - Python environment with required dependencies

Data Directory Structure:
    sae_data_directory/
    ├── explanations/     # Feature explanations (batch-*.jsonl.gz)
    ├── activations/      # Activation patterns (batch-*.jsonl.gz)
    └── features/         # Feature statistics (batch-*.jsonl.gz)
"""

import sys
import subprocess
import json
import glob
import gzip
import psycopg2
from typing import List
from pathlib import Path

class NeuronpediaDeployer:
    def __init__(self, sae_data_dir: str, webapp_dir: str = None):
        self.sae_data_dir = Path(sae_data_dir)
        self.webapp_dir = Path(webapp_dir) if webapp_dir else Path("/Users/akash/Desktop/code/neuronpedia/apps/webapp")
        self.script_dir = Path(__file__).parent  # Directory containing this script and other realm scripts
        
        # Validate paths
        if not self.sae_data_dir.exists():
            raise ValueError(f"SAE data directory not found: {self.sae_data_dir}")
        if not self.webapp_dir.exists():
            raise ValueError(f"Webapp directory not found: {self.webapp_dir}")
    
    def run_command(self, cmd: str, cwd: str = None, description: str = None):
        """Run a shell command and handle errors."""
        if description:
            print(f"🔄 {description}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                print(f"Error: {result.stderr}")
                return False
            
            if description:
                print(f"✅ {description} - Complete")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"⏱️ Command timed out: {cmd}")
            return False
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False
    
    def check_postgres(self):
        """Check if PostgreSQL is running and accessible."""
        print("🔍 Checking PostgreSQL connection...")
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="postgres",
                user="postgres",
                password="postgres",
                connect_timeout=10
            )
            conn.close()
            print("✅ PostgreSQL connection successful")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            print("Please ensure PostgreSQL is running on localhost:5432 with postgres/postgres")
            return False
    
    def setup_database(self):
        """Set up the database schema."""
        print("🗄️ Setting up database schema...")
        
        # Run database migrations
        success = self.run_command(
            "npx prisma db push",
            cwd=str(self.webapp_dir),
            description="Running Prisma database migrations"
        )
        
        if not success:
            print("❌ Database setup failed")
            return False
        
        print("✅ Database schema setup complete")
        return True
    
    def build_webapp(self):
        """Build the webapp."""
        print("🏗️ Building webapp...")
        
        # Install dependencies
        success = self.run_command(
            "npm install",
            cwd=str(self.webapp_dir),
            description="Installing webapp dependencies"
        )
        
        if not success:
            return False
        
        # Build webapp
        success = self.run_command(
            "npm run build:localhost",
            cwd=str(self.webapp_dir),
            description="Building webapp for localhost"
        )
        
        if not success:
            return False
        
        print("✅ Webapp build complete")
        return True
    
    def load_json_data(self, file_path: str) -> List[dict]:
        """Load data from JSONL file (gzipped or not)."""
        data = []
        
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        return data
    
    def get_model_info(self) -> tuple:
        """Extract model info from data files."""
        # Try explanations first
        explanations_dir = self.sae_data_dir / "explanations"
        if explanations_dir.exists():
            batch_files = sorted(glob.glob(str(explanations_dir / "batch-*.jsonl*")))
            if batch_files:
                data = self.load_json_data(batch_files[0])
                if data:
                    return data[0]['modelId'], 21  # Assuming 21 layers for Llama 3.1 8B
        
        # Try features
        features_dir = self.sae_data_dir / "features"
        if features_dir.exists():
            batch_files = sorted(glob.glob(str(features_dir / "batch-*.jsonl*")))
            if batch_files:
                data = self.load_json_data(batch_files[0])
                if data:
                    return data[0]['modelId'], 21
        
        # Try activations
        activations_dir = self.sae_data_dir / "activations"
        if activations_dir.exists():
            batch_files = sorted(glob.glob(str(activations_dir / "batch-*.jsonl*")))
            if batch_files:
                data = self.load_json_data(batch_files[0])
                if data:
                    return data[0]['modelId'], 21
        
        raise ValueError("Could not determine model info from data files")
    
    def upload_explanations_and_base_setup(self):
        """Upload explanations and set up basic database structure."""
        print("📝 Uploading explanations and setting up base structure...")
        
        explanations_dir = self.sae_data_dir / "explanations"
        if not explanations_dir.exists():
            print(f"⚠️ No explanations directory found at {explanations_dir}")
            return False
        
        # Use the existing upload script
        cmd = f'python "{self.script_dir}/upload_explanations.py" "{explanations_dir}" autointerp-sae'
        success = self.run_command(
            cmd,
            description="Uploading explanations and creating base structure"
        )
        
        return success
    
    def upload_activations(self):
        """Upload activation data."""
        print("🎯 Uploading activation data...")
        
        activations_dir = self.sae_data_dir / "activations"
        if not activations_dir.exists():
            print(f"⚠️ No activations directory found at {activations_dir}")
            return False
        
        # Use the streaming upload script
        cmd = f'python "{self.script_dir}/upload_activations.py" "{activations_dir}" autointerp-sae --skip-pruning'
        
        print("⚠️ This may take 20-30 minutes for large datasets...")
        success = self.run_command(
            cmd,
            description="Uploading all activation data"
        )
        
        return success
    
    def update_feature_statistics(self):
        """Update neurons with rich feature statistics."""
        print("📊 Updating feature statistics...")
        
        features_dir = self.sae_data_dir / "features"
        if not features_dir.exists():
            print(f"⚠️ No features directory found at {features_dir}")
            return False
        
        # Use the feature update script
        cmd = f'python "{self.script_dir}/upload_feature_stats.py" "{features_dir}" autointerp-sae'
        
        print("⚠️ This may take 10-15 minutes for large datasets...")
        success = self.run_command(
            cmd,
            description="Updating all feature statistics"
        )
        
        return success
    
    def fix_max_act_approx(self):
        """Fix maxActApprox values for proper feature display."""
        print("🔧 Fixing maxActApprox values...")
        
        cmd = f'python "{self.script_dir}/fix_max_act.py"'
        success = self.run_command(
            cmd,
            description="Fixing maxActApprox values"
        )
        
        return success
    
    def start_webapp(self):
        """Start the webapp in development mode."""
        print("🚀 Starting webapp...")
        print("📋 The webapp will start on http://localhost:3000")
        print("📋 Use Ctrl+C to stop the server")
        print("📋 After starting, you can view your features at:")
        
        try:
            model_id, _ = self.get_model_info()
            clean_model_id = model_id.replace('/', '-').replace('_', '-').lower()
            print(f"   http://localhost:3000/{clean_model_id}")
            print(f"   http://localhost:3000/{clean_model_id}/20-autointerp-sae/311")
        except:
            print("   http://localhost:3000")
        
        print("\n" + "="*50)
        
        # Start the development server (this will run indefinitely)
        subprocess.run(
            "npm run dev:localhost",
            shell=True,
            cwd=str(self.webapp_dir)
        )
    
    def deploy(self):
        """Run the complete deployment process."""
        print("🚀 Starting Neuronpedia Local Deployment")
        print("="*50)
        
        # Step 1: Check prerequisites
        if not self.check_postgres():
            return False
        
        # Step 2: Set up database
        if not self.setup_database():
            return False
        
        # Step 3: Build webapp
        if not self.build_webapp():
            return False
        
        # Step 4: Upload explanations and basic structure
        if not self.upload_explanations_and_base_setup():
            return False
        
        # Step 5: Upload activations
        if not self.upload_activations():
            print("⚠️ Activation upload failed, continuing...")
        
        # Step 6: Update feature statistics
        if not self.update_feature_statistics():
            print("⚠️ Feature statistics update failed, continuing...")
        
        # Step 7: Fix maxActApprox values
        if not self.fix_max_act_approx():
            print("⚠️ maxActApprox fix failed, continuing...")
        
        print("\n🎉 Deployment Complete!")
        print("="*50)
        
        # Step 8: Start webapp
        self.start_webapp()

def main():
    if len(sys.argv) < 2:
        print("Usage: python deploy_neuronpedia_local.py <sae_data_directory>")
        print("Example: python deploy_neuronpedia_local.py /Users/akash/Desktop/code/20-realm-l20r-8x")
        print()
        print("Prerequisites:")
        print("- PostgreSQL running on localhost:5432 with postgres/postgres")
        print("- Node.js and npm installed")
        print("- Python with required dependencies (psycopg2)")
        sys.exit(1)
    
    sae_data_dir = sys.argv[1]
    
    try:
        deployer = NeuronpediaDeployer(sae_data_dir)
        deployer.deploy()
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()