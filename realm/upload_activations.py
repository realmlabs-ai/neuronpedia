#!/usr/bin/env python3
"""
Neuronpedia Activations Streaming Upload Script

Efficiently uploads large volumes of SAE activation data using a streaming approach.
Processes activation files sequentially to minimize memory usage while maintaining
high throughput for large datasets.

This script supports optional pruning to keep only the top N activations per feature,
or can upload all activations when pruning is disabled.

Usage:
    python upload_activations_streaming.py <activations_directory> [top_n] [source_set_name] [--skip-pruning]

Arguments:
    activations_directory: Path to directory containing activation batch files
    top_n: Number of top activations to keep per feature (default: 20)
    source_set_name: Optional source set name (default: "autointerp-sae")
    --skip-pruning: Upload all activations without pruning

Examples:
    # Upload top 20 activations per feature
    python upload_activations_streaming.py /path/to/activations 20

    # Upload all activations without pruning
    python upload_activations_streaming.py /path/to/activations --skip-pruning

Input Format:
    Directory should contain batch-*.jsonl or batch-*.jsonl.gz files with
    activation data in JSONL format.
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

def load_activations_from_jsonl(file_path: str) -> List[dict]:
    """Load activations from a JSONL file (gzipped or not)."""
    activations = []
    
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                for line in f:
                    if line.strip():
                        activations.append(json.loads(line))
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        activations.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return activations

def extract_layer_from_source(layer_field: str) -> int:
    """Extract layer number from source identifier like '20-realm-l20r-8x' -> 20"""
    if '-' in layer_field:
        return int(layer_field.split('-')[0])
    else:
        return int(layer_field)

def clear_existing_activations(conn, model_id: str, source_set_name: str):
    """Clear existing activations for this model/source to avoid duplicates."""
    cursor = conn.cursor()
    
    try:
        print(f"🧹 Clearing existing activations for {model_id}/{source_set_name}...")
        
        cursor.execute('''
            DELETE FROM "Activation" 
            WHERE "modelId" = %s AND layer LIKE %s
        ''', (model_id, f'%-{source_set_name}'))
        
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"✅ Deleted {deleted_count} existing activations")
        
    except Exception as e:
        print(f"Error clearing activations: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def stream_upload_activations(conn, activations_dir: str, model_id: str, source_set_name: str, user_id: str):
    """Stream upload activations from files directly to database."""
    cursor = conn.cursor()
    total_inserted = 0
    batch_files = sorted(glob.glob(os.path.join(activations_dir, "batch-*.jsonl*")))
    
    print(f"📤 Streaming {len(batch_files)} activation files to database...")
    
    try:
        for i, batch_file in enumerate(batch_files):
            if i % 10 == 0:
                print(f"  Processing file {i+1}/{len(batch_files)}: {os.path.basename(batch_file)}")
            
            activations = load_activations_from_jsonl(batch_file)
            
            for j, act_data in enumerate(activations):
                layer = extract_layer_from_source(act_data['layer'])
                source_id = f"{layer}-{source_set_name}"
                feature_index = str(act_data['index'])
                
                # Generate unique activation ID using file and position indices
                activation_id = f"act_{model_id}_{layer}_{feature_index}_{i}_{j}"
                
                # Check if corresponding neuron exists before inserting activation
                cursor.execute('''
                    SELECT 1 FROM "Neuron" 
                    WHERE "modelId" = %s AND layer = %s AND index = %s
                ''', (model_id, source_id, feature_index))
                
                if not cursor.fetchone():
                    # Skip this activation if no corresponding neuron exists
                    continue
                
                try:
                    cursor.execute('''
                        INSERT INTO "Activation" (
                            id, "modelId", layer, index, "creatorId", 
                            tokens, values, "maxValue", "minValue", 
                            "maxValueTokenIndex", "createdAt"
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        ON CONFLICT (id) DO NOTHING
                    ''', (
                        activation_id,
                        model_id,
                        source_id,
                        feature_index,
                        user_id,
                        act_data['tokens'],
                        act_data['values'],
                        act_data['maxValue'],
                        act_data['minValue'],
                        act_data.get('maxValueTokenIndex', 0)
                    ))
                    
                    if cursor.rowcount > 0:
                        total_inserted += 1
                        
                except Exception as e:
                    print(f"Error inserting activation: {e}")
                    print(f"  File: {batch_file}")
                    print(f"  Feature: {feature_index}")
                    conn.rollback()
                    raise
            
            # Commit every batch file
            conn.commit()
            
            if (i + 1) % 50 == 0:
                print(f"    Inserted {total_inserted} activations so far...")
    
    except Exception as e:
        print(f"Error during streaming upload: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
    
    return total_inserted

def prune_to_top_n_per_feature(conn, model_id: str, source_set_name: str, top_n: int):
    """Keep only top N activations per feature, delete the rest."""
    cursor = conn.cursor()
    
    try:
        print(f"✂️  Pruning to top {top_n} activations per feature...")
        
        # Find all source IDs for this model/source set
        cursor.execute('''
            SELECT DISTINCT layer FROM "Activation" 
            WHERE "modelId" = %s AND layer LIKE %s
        ''', (model_id, f'%-{source_set_name}'))
        
        source_ids = [row[0] for row in cursor.fetchall()]
        
        total_deleted = 0
        for source_id in source_ids:
            print(f"  Pruning layer {source_id}...")
            
            # Delete activations that are not in the top N for each feature
            cursor.execute('''
                DELETE FROM "Activation" a1 
                WHERE a1."modelId" = %s 
                AND a1.layer = %s
                AND (
                    SELECT COUNT(*) 
                    FROM "Activation" a2 
                    WHERE a2."modelId" = a1."modelId" 
                    AND a2.layer = a1.layer 
                    AND a2.index = a1.index 
                    AND a2."maxValue" > a1."maxValue"
                ) >= %s
            ''', (model_id, source_id, top_n))
            
            deleted_count = cursor.rowcount
            total_deleted += deleted_count
            print(f"    Deleted {deleted_count} excess activations")
            
            conn.commit()
        
        print(f"✅ Pruned {total_deleted} activations, keeping top {top_n} per feature")
        
        # Show final counts
        cursor.execute('''
            SELECT layer, index, COUNT(*) as activation_count
            FROM "Activation" 
            WHERE "modelId" = %s AND layer LIKE %s
            GROUP BY layer, index 
            ORDER BY layer, CAST(index AS INTEGER)
            LIMIT 5
        ''', (model_id, f'%-{source_set_name}'))
        
        sample_counts = cursor.fetchall()
        print(f"\nSample activation counts:")
        for layer, index, count in sample_counts:
            print(f"  {layer}/{index}: {count} activations")
        
    except Exception as e:
        print(f"Error during pruning: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def streaming_upload_main(activations_dir: str, top_n: int = 20, source_set_name: str = None, skip_pruning: bool = False):
    """Main function for streaming upload."""
    
    if source_set_name is None:
        source_set_name = "autointerp-sae"
    
    # Get model info from first file
    batch_files = sorted(glob.glob(os.path.join(activations_dir, "batch-*.jsonl*")))
    if not batch_files:
        print(f"❌ No batch files found in {activations_dir}")
        return
    
    first_activations = load_activations_from_jsonl(batch_files[0])
    if not first_activations:
        print(f"❌ No activations in first file")
        return
    
    model_id = first_activations[0]['modelId']
    clean_model_id = model_id.replace('/', '-').replace('_', '-').lower()
    print(f"Detected model: {model_id} -> {clean_model_id}")
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        user_id = 'akash-realm-local'
        
        # Ask about clearing existing
        response = input(f"\n⚠️  Clear existing activations for {clean_model_id}/{source_set_name}? (y/N): ")
        if response.lower() in ['y', 'yes']:
            clear_existing_activations(conn, clean_model_id, source_set_name)
        
        # Stream upload all activations
        print(f"\n🚀 Starting streaming upload...")
        total_inserted = stream_upload_activations(conn, activations_dir, clean_model_id, source_set_name, user_id)
        
        print(f"\n📊 Uploaded {total_inserted} total activations")
        
        # Optionally prune to top N per feature
        if not skip_pruning:
            prune_to_top_n_per_feature(conn, clean_model_id, source_set_name, top_n)
        else:
            print(f"⏭️  Skipping pruning - keeping all {total_inserted} activations")
        
        print(f"\n🎉 Streaming upload complete!")
        print(f"📊 View your enhanced features at:")
        print(f"   http://localhost:3000/{clean_model_id}")
        print(f"   Example: http://localhost:3000/{clean_model_id}/20-{source_set_name}/311")
    
    finally:
        conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_activations_streaming.py <activations_dir> [top_n] [source_set_name] [--skip-pruning]")
        print("Example: python upload_activations_streaming.py /Users/akash/Desktop/code/20-realm-l20r-8x/activations 20")
        print("Example (no pruning): python upload_activations_streaming.py /Users/akash/Desktop/code/20-realm-l20r-8x/activations --skip-pruning")
        sys.exit(1)
    
    activations_dir = sys.argv[1]
    skip_pruning = '--skip-pruning' in sys.argv
    
    # Parse arguments based on whether we're skipping pruning
    if skip_pruning:
        # When skipping pruning, we don't need top_n
        top_n = None
        # Find source_set_name (any non-flag argument after activations_dir)
        source_set_name = None
        for arg in sys.argv[2:]:
            if not arg.startswith('--'):
                source_set_name = arg
                break
    else:
        # When pruning, we need top_n
        top_n = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 20
        source_set_name = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
    
    if not os.path.exists(activations_dir):
        print(f"❌ Directory not found: {activations_dir}")
        sys.exit(1)
    
    print(f"Streaming Activation Upload")
    print(f"===========================")
    print(f"Activations dir: {activations_dir}")
    if not skip_pruning:
        print(f"Top N per feature: {top_n}")
    print(f"Source set: {source_set_name or 'autointerp-sae'}")
    print(f"Skip pruning: {skip_pruning}")
    print()
    
    streaming_upload_main(activations_dir, top_n or 20, source_set_name, skip_pruning)

if __name__ == "__main__":
    main()