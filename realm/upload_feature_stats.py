#!/usr/bin/env python3
"""
Neuronpedia Feature Statistics Update Script

Updates existing neuron records with comprehensive feature statistics including
positive/negative logits, activation density, histograms, and correlation data.

This script enhances basic neuron records with rich statistical information
required for full Neuronpedia dashboard functionality, including the detailed
feature analysis panels and visualization components.

Usage:
    python update_feature_stats.py <features_directory> [source_set_name]

Arguments:
    features_directory: Path to directory containing feature statistics files
    source_set_name: Optional source set name (default: "autointerp-sae")

Example:
    python update_feature_stats.py /path/to/features autointerp-sae

Input Format:
    Directory should contain batch-*.jsonl or batch-*.jsonl.gz files with
    complete feature statistics in JSONL format.

Updated Fields:
    - pos_str, pos_values: Top positive activating tokens and values
    - neg_str, neg_values: Top negative activating tokens and values
    - frac_nonzero: Feature activation density percentage
    - freq_hist_data_*: Frequency distribution histograms
    - logits_hist_data_*: Logits distribution histograms
    - correlation data: Feature and neuron correlation matrices
"""

import os
import sys
import json
import glob
import gzip
import psycopg2
from typing import List

def connect_to_db():
    """Connect to the local PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="postgres"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def load_features_from_jsonl(file_path: str) -> List[dict]:
    """Load features from a JSONL file (gzipped or not)."""
    features = []
    
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                for line in f:
                    if line.strip():
                        features.append(json.loads(line))
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        features.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return features

def extract_layer_from_source(layer_field: str) -> int:
    """Extract layer number from source identifier like '20-realm-l20r-8x' -> 20"""
    if '-' in layer_field:
        return int(layer_field.split('-')[0])
    else:
        return int(layer_field)

def update_neuron_with_features(conn, feature_data: dict, model_id: str, source_set_name: str):
    """Update a single neuron with feature statistics."""
    cursor = conn.cursor()
    
    try:
        layer = extract_layer_from_source(feature_data['layer'])
        source_id = f"{layer}-{source_set_name}"
        feature_index = str(feature_data['index'])
        
        # Update neuron with all the rich feature data
        cursor.execute('''
            UPDATE "Neuron" SET
                "maxActApprox" = %s,
                "hasVector" = %s,
                vector = %s,
                "vectorDefaultSteerStrength" = %s,
                "vectorLabel" = %s,
                "topkCosSimIndices" = %s,
                "topkCosSimValues" = %s,
                neuron_alignment_indices = %s,
                neuron_alignment_values = %s,
                neuron_alignment_l1 = %s,
                correlated_neurons_indices = %s,
                correlated_neurons_pearson = %s,
                correlated_neurons_l1 = %s,
                correlated_features_indices = %s,
                correlated_features_pearson = %s,
                correlated_features_l1 = %s,
                neg_str = %s,
                neg_values = %s,
                pos_str = %s,
                pos_values = %s,
                frac_nonzero = %s,
                freq_hist_data_bar_heights = %s,
                freq_hist_data_bar_values = %s,
                logits_hist_data_bar_heights = %s,
                logits_hist_data_bar_values = %s,
                decoder_weights_dist = %s,
                "hookName" = %s
            WHERE "modelId" = %s AND layer = %s AND index = %s
        ''', (
            feature_data.get('maxActApprox', 0),
            feature_data.get('hasVector', False),
            feature_data.get('vector', []),
            feature_data.get('vectorDefaultSteerStrength', 10),
            feature_data.get('vectorLabel'),
            feature_data.get('topkCosSimIndices', []),
            feature_data.get('topkCosSimValues', []),
            feature_data.get('neuron_alignment_indices', []),
            feature_data.get('neuron_alignment_values', []),
            feature_data.get('neuron_alignment_l1', []),
            feature_data.get('correlated_neurons_indices', []),
            feature_data.get('correlated_neurons_pearson', []),
            feature_data.get('correlated_neurons_l1', []),
            feature_data.get('correlated_features_indices', []),
            feature_data.get('correlated_features_pearson', []),
            feature_data.get('correlated_features_l1', []),
            feature_data.get('neg_str', []),
            feature_data.get('neg_values', []),
            feature_data.get('pos_str', []),
            feature_data.get('pos_values', []),
            feature_data.get('frac_nonzero', 0),
            feature_data.get('freq_hist_data_bar_heights', []),
            feature_data.get('freq_hist_data_bar_values', []),
            feature_data.get('logits_hist_data_bar_heights', []),
            feature_data.get('logits_hist_data_bar_values', []),
            feature_data.get('decoder_weights_dist', []),
            feature_data.get('hookName'),
            model_id,
            source_id,
            feature_index
        ))
        
        return cursor.rowcount > 0
        
    except Exception as e:
        print(f"Error updating neuron {feature_index}: {e}")
        return False
    finally:
        cursor.close()

def update_feature_statistics(features_dir: str, source_set_name: str = None):
    """Update existing neurons with feature statistics."""
    
    if source_set_name is None:
        source_set_name = "autointerp-sae"
    
    # Get model info from first file
    batch_files = sorted(glob.glob(os.path.join(features_dir, "batch-*.jsonl*")))
    if not batch_files:
        print(f"❌ No batch files found in {features_dir}")
        return
    
    first_features = load_features_from_jsonl(batch_files[0])
    if not first_features:
        print(f"❌ No features in first file")
        return
    
    model_id = first_features[0]['modelId']
    clean_model_id = model_id.replace('/', '-').replace('_', '-').lower()
    print(f"Detected model: {model_id} -> {clean_model_id}")
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        total_updated = 0
        
        print(f"🔄 Updating feature statistics for {len(batch_files)} batch files...")
        
        for i, batch_file in enumerate(batch_files):
            if i % 10 == 0:
                print(f"  Processing file {i+1}/{len(batch_files)}: {os.path.basename(batch_file)}")
            
            features = load_features_from_jsonl(batch_file)
            
            for feature_data in features:
                if update_neuron_with_features(conn, feature_data, clean_model_id, source_set_name):
                    total_updated += 1
            
            # Commit after each batch file
            conn.commit()
            
            if (i + 1) % 50 == 0:
                print(f"    Updated {total_updated} neurons so far...")
        
        print(f"\n✅ Updated {total_updated} neurons with feature statistics!")
        print(f"🎉 Your webapp should now show positive/negative logits and activation density!")
        print(f"📊 View at: http://localhost:3000/{clean_model_id}/20-{source_set_name}/311")
        
    except Exception as e:
        print(f"Error updating features: {e}")
        conn.rollback()
    finally:
        conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_feature_stats.py <features_dir> [source_set_name]")
        print("Example: python update_feature_stats.py /Users/akash/Desktop/code/20-realm-l20r-8x/features")
        sys.exit(1)
    
    features_dir = sys.argv[1]
    source_set_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(features_dir):
        print(f"❌ Directory not found: {features_dir}")
        sys.exit(1)
    
    print(f"Update Feature Statistics")
    print(f"=========================")
    print(f"Features dir: {features_dir}")
    print(f"Source set: {source_set_name or 'autointerp-sae'}")
    print()
    
    update_feature_statistics(features_dir, source_set_name)

if __name__ == "__main__":
    main()