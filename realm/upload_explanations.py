#!/usr/bin/env python3
"""
Neuronpedia Explanations Upload Script

Uploads SAE feature explanations to a local Neuronpedia database instance.
Handles complete database setup including users, models, sources, and explanations.

This script automatically creates all required database entities and dependencies,
making it suitable for initial database population or adding new explanation sets.

Usage:
    python upload_explanations_complete.py <explanations_directory> [source_set_name]

Arguments:
    explanations_directory: Path to directory containing explanation batch files
    source_set_name: Optional name for the source set (default: "autointerp-sae")

Example:
    python upload_explanations_complete.py /path/to/explanations autointerp-sae

Input Format:
    Directory should contain batch-*.jsonl or batch-*.jsonl.gz files with
    explanation data in JSONL format.
"""

import os
import sys
import json
import glob
import gzip
import psycopg2
from typing import List, Dict

def load_explanations_from_jsonl(file_path: str) -> List[dict]:
    """Load explanations from a JSONL file (gzipped or not)."""
    explanations = []
    
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                for line in f:
                    if line.strip():
                        explanations.append(json.loads(line))
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        explanations.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return explanations

def extract_layer_from_source(layer_field: str) -> int:
    """Extract layer number from source identifier like '20-realm-l20r-8x' -> 20"""
    if '-' in layer_field:
        return int(layer_field.split('-')[0])
    else:
        return int(layer_field)

def get_model_info_from_explanations(explanations_dir: str) -> tuple:
    """Extract model info from explanation files."""
    batch_files = sorted(glob.glob(os.path.join(explanations_dir, "batch-*.jsonl*")))
    
    if not batch_files:
        raise ValueError(f"No batch-*.jsonl files found in {explanations_dir}")
    
    first_file = batch_files[0]
    explanations = load_explanations_from_jsonl(first_file)
    
    if not explanations:
        raise ValueError(f"No explanations found in {first_file}")
    
    first_exp = explanations[0]
    model_id = first_exp['modelId']
    
    # Get all unique layers
    layers = set()
    for batch_file in batch_files[:3]:
        batch_explanations = load_explanations_from_jsonl(batch_file)
        for exp in batch_explanations:
            layer_num = extract_layer_from_source(exp['layer'])
            layers.add(layer_num)
    
    max_layer = max(layers) if layers else 0
    return model_id, max_layer + 1

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

def ensure_user_exists(conn, user_id: str = 'akash-realm-local'):
    """Ensure the user exists in the database."""
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT id FROM "User" WHERE id = %s', (user_id,))
        if cursor.fetchone():
            print(f"User {user_id} already exists")
            return user_id
        
        # Create user if doesn't exist
        cursor.execute('''
            INSERT INTO "User" (id, name, email, "emailUnsubscribeCode")
            VALUES (%s, %s, %s, %s)
        ''', (user_id, 'Akash Realm Local', 'akash@realm.local', f'unsubscribe-{user_id}'))
        
        conn.commit()
        print(f"Created user: {user_id}")
        return user_id
        
    except Exception as e:
        print(f"FATAL: Error with user: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()

def ensure_model_exists(conn, model_id: str, num_layers: int, user_id: str):
    """Ensure the model exists in the database."""
    cursor = conn.cursor()
    
    try:
        # Clean model ID first
        clean_model_id = model_id.replace('/', '-').replace('_', '-').lower()
        
        # Check if clean model ID exists
        cursor.execute('SELECT id FROM "Model" WHERE id = %s', (clean_model_id,))
        if cursor.fetchone():
            print(f"Model {clean_model_id} already exists")
            return clean_model_id
        
        # Create model with clean ID - assume standard model dimensions
        cursor.execute('''
            INSERT INTO "Model" (
                id, "displayName", "creatorId", layers, "neuronsPerLayer", 
                "owner", visibility, "createdAt", "updatedAt"
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        ''', (clean_model_id, f"{model_id} (Local)", user_id, num_layers, 4096, user_id, 'PUBLIC'))
        
        conn.commit()
        print(f"Created model: {clean_model_id} (cleaned from {model_id}) with {num_layers} layers")
        return clean_model_id
        
    except Exception as e:
        print(f"FATAL: Error creating model: {e}")
        print(f"Trying to create model with ID: {model_id}")
        print(f"User ID: {user_id}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()

def ensure_source_set_exists(conn, model_id: str, source_set_name: str, num_layers: int, user_id: str):
    """Ensure source set and sources exist."""
    cursor = conn.cursor()
    
    try:
        # Check if source set exists
        cursor.execute('''
            SELECT name FROM "SourceSet" WHERE "modelId" = %s AND name = %s
        ''', (model_id, source_set_name))
        
        if cursor.fetchone():
            print(f"Source set {source_set_name} already exists")
        else:
            # Create source set with all required fields
            cursor.execute('''
                INSERT INTO "SourceSet" (
                    "modelId", name, description, "creatorName", "creatorId", visibility, "createdAt"
                )
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ''', (
                model_id, 
                source_set_name, 
                f"Auto-interpreted SAE features for {model_id}",
                'akash-realm-local',
                user_id,
                'PUBLIC'
            ))
            print(f"Created source set: {source_set_name}")
        
        # Create sources for each layer
        for layer in range(num_layers):
            source_id = f"{layer}-{source_set_name}"
            
            cursor.execute('SELECT id FROM "Source" WHERE id = %s', (source_id,))
            if cursor.fetchone():
                continue
                
            cursor.execute('''
                INSERT INTO "Source" (
                    id, "modelId", "setName", "creatorId", "createdAt"
                )
                VALUES (%s, %s, %s, %s, NOW())
            ''', (source_id, model_id, source_set_name, user_id))
            
        conn.commit()
        print(f"Ensured sources exist for {num_layers} layers")
        
    except Exception as e:
        print(f"FATAL: Error with source set: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()

def ensure_explanation_types_exist(conn, user_id: str):
    """Ensure required explanation types exist."""
    cursor = conn.cursor()
    
    explanation_types = [
        ('np_max-act-logits', 'Max Activation with Logits', 'Neuronpedia max activation with logits explanation'),
        ('oai_token-act-pair', 'Token Activation Pair', 'OpenAI token activation pair explanation'),
        ('imported', 'Imported', 'Imported explanation from external source')
    ]
    
    try:
        for type_name, display_name, description in explanation_types:
            cursor.execute('''
                INSERT INTO "ExplanationType" (
                    name, "displayName", description, "creatorId", "creatorName", "createdAt", "updatedAt"
                )
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (name) DO NOTHING
            ''', (type_name, display_name, description, user_id, 'system'))
        
        conn.commit()
        print("Ensured explanation types exist")
        
    except Exception as e:
        print(f"FATAL: Error creating explanation types: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()

def ensure_explanation_model_types_exist(conn):
    """Ensure required explanation model types exist."""
    cursor = conn.cursor()
    
    model_types = [
        ('gpt-4o-mini', 'GPT-4o Mini', 'OpenAI GPT-4o Mini model'),
        ('unknown', 'Unknown', 'Unknown explanation model')
    ]
    
    try:
        for model_name, display_name, description in model_types:
            cursor.execute('''
                INSERT INTO "ExplanationModelType" (
                    name, "displayName", description, "creatorName", "createdAt", "updatedAt"
                )
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (name) DO NOTHING
            ''', (model_name, display_name, description, 'system'))
        
        conn.commit()
        print("Ensured explanation model types exist")
        
    except Exception as e:
        print(f"FATAL: Error creating explanation model types: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()

def ensure_neurons_exist(conn, explanations_by_layer: Dict[int, List], model_id: str, source_set_name: str, user_id: str):
    """Create Neuron records for each explanation."""
    cursor = conn.cursor()
    
    try:
        for layer, explanations in explanations_by_layer.items():
            source_id = f"{layer}-{source_set_name}"
            
            for exp_data in explanations:
                # Convert index to string to match database schema
                neuron_index = str(exp_data['index'])
                
                # Check if neuron already exists
                cursor.execute('''
                    SELECT "modelId" FROM "Neuron" 
                    WHERE "modelId" = %s AND layer = %s AND index = %s
                ''', (model_id, source_id, neuron_index))
                
                if cursor.fetchone():
                    continue
                
                # Create neuron
                cursor.execute('''
                    INSERT INTO "Neuron" (
                        "modelId", layer, index, "sourceSetName", "creatorId", "createdAt"
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT ("modelId", layer, index) DO NOTHING
                ''', (model_id, source_id, neuron_index, source_set_name, user_id))
        
        conn.commit()
        print("Ensured neurons exist for all explanations")
        
    except Exception as e:
        print(f"FATAL: Error creating neurons: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()

def insert_explanations(conn, explanations_by_layer: Dict[int, List], model_id: str, source_set_name: str, user_id: str):
    """Insert explanations into database."""
    cursor = conn.cursor()
    total_inserted = 0
    
    try:
        for layer, explanations in explanations_by_layer.items():
            print(f"Inserting {len(explanations)} explanations for layer {layer}...")
            source_id = f"{layer}-{source_set_name}"
            
            batch_size = 100
            for i in range(0, len(explanations), batch_size):
                batch = explanations[i:i+batch_size]
                
                for exp_data in batch:
                    # Convert index to string to match database schema
                    neuron_index = str(exp_data['index'])
                    explanation_id = f"local_{model_id}_{layer}_{neuron_index}"
                    
                    try:
                        cursor.execute('''
                            INSERT INTO "Explanation" (
                                id, "modelId", layer, index, description, 
                                "typeName", "explanationModelName", 
                                "authorId", "createdAt", "updatedAt"
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                            ON CONFLICT (id) DO NOTHING
                        ''', (
                            explanation_id,
                            model_id,
                            source_id,  # Use source_id (e.g., "20-autointerp-sae") instead of layer number
                            neuron_index,  # Use string version of index
                            exp_data['description'],
                            exp_data.get('typeName', 'imported'),
                            exp_data.get('explanationModelName', 'unknown'),
                            user_id
                        ))
                        
                        if cursor.rowcount > 0:
                            total_inserted += 1
                            
                    except Exception as e:
                        print(f"FATAL: Error inserting explanation {exp_data['index']}: {e}")
                        conn.rollback()
                        sys.exit(1)
                
                # Commit every batch
                conn.commit()
            
            print(f"  Completed layer {layer}")
    
    except Exception as e:
        print(f"FATAL: Error inserting explanations: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()
    
    return total_inserted

def upload_activations(conn, activations_by_layer: Dict[int, List], model_id: str, source_set_name: str, user_id: str):
    """Insert activation data into database."""
    cursor = conn.cursor()
    total_inserted = 0
    
    try:
        for layer, activations in activations_by_layer.items():
            print(f"Inserting {len(activations)} activations for layer {layer}...")
            source_id = f"{layer}-{source_set_name}"
            
            batch_size = 100
            for i in range(0, len(activations), batch_size):
                batch = activations[i:i+batch_size]
                
                for act_data in batch:
                    # Convert index to string to match database schema
                    neuron_index = str(act_data['index'])
                    activation_id = f"act_{model_id}_{layer}_{neuron_index}"
                    
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
                            neuron_index,  # Use string version of index
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
                        print(f"FATAL: Error inserting activation {act_data['index']}: {e}")
                        conn.rollback()
                        sys.exit(1)
                
                # Commit every batch
                conn.commit()
            
            print(f"  Completed layer {layer}")
    
    except Exception as e:
        print(f"FATAL: Error inserting activations: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cursor.close()
    
    return total_inserted

def upload_explanations_complete(explanations_dir: str, source_set_name: str = None, upload_activations_too: bool = False):
    """Complete upload with all dependencies handled."""
    
    # Get model info
    try:
        model_id, num_layers = get_model_info_from_explanations(explanations_dir)
        print(f"Detected model: {model_id} with {num_layers} layers")
    except Exception as e:
        print(f"Error detecting model info: {e}")
        return
    
    if source_set_name is None:
        source_set_name = "autointerp-sae"
    
    # Connect to database
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        # Ensure all dependencies exist
        print("\n🔧 Setting up database dependencies...")
        user_id = ensure_user_exists(conn)
        ensure_explanation_types_exist(conn, user_id)
        ensure_explanation_model_types_exist(conn)
        clean_model_id = ensure_model_exists(conn, model_id, num_layers, user_id)
            
        # Use clean model ID for everything else
        if clean_model_id != model_id:
            print(f"Using cleaned model ID: {clean_model_id}")
            model_id = clean_model_id
            
        ensure_source_set_exists(conn, model_id, source_set_name, num_layers, user_id)
        
        # Load all explanations
        print("\n📄 Loading explanation files...")
        batch_files = sorted(glob.glob(os.path.join(explanations_dir, "batch-*.jsonl*")))
        explanations_by_layer = {}
        
        for batch_file in batch_files:
            print(f"Processing {os.path.basename(batch_file)}...")
            explanations_data = load_explanations_from_jsonl(batch_file)
            
            for exp_data in explanations_data:
                layer = extract_layer_from_source(exp_data['layer'])
                if layer not in explanations_by_layer:
                    explanations_by_layer[layer] = []
                explanations_by_layer[layer].append(exp_data)
        
        # Load activations if requested
        activations_by_layer = {}
        if upload_activations_too:
            print("\n📊 Loading activation files...")
            # Look for activation files in parent directory
            activations_dir = os.path.join(os.path.dirname(explanations_dir), "activations")
            if os.path.exists(activations_dir):
                activation_files = sorted(glob.glob(os.path.join(activations_dir, "batch-*.jsonl*")))
                
                for activation_file in activation_files:
                    print(f"Processing {os.path.basename(activation_file)}...")
                    activations_data = load_explanations_from_jsonl(activation_file)  # Same loader works for activations
                    
                    for act_data in activations_data:
                        layer = extract_layer_from_source(act_data['layer'])
                        if layer not in activations_by_layer:
                            activations_by_layer[layer] = []
                        activations_by_layer[layer].append(act_data)
                        
                print(f"Loaded activations for {len(activations_by_layer)} layers")
            else:
                print(f"⚠️  No activations directory found at {activations_dir}, skipping activations")
                upload_activations_too = False
        
        # Create neurons for explanations
        print("\n🧠 Creating neurons for explanations...")
        ensure_neurons_exist(conn, explanations_by_layer, model_id, source_set_name, user_id)
        
        # Create neurons for activations (if they don't already exist)
        if upload_activations_too and activations_by_layer:
            print("\n🧠 Creating neurons for activations...")
            ensure_neurons_exist(conn, activations_by_layer, model_id, source_set_name, user_id)
        
        # Insert explanations
        print("\n📊 Inserting explanations...")
        total_explanations_inserted = insert_explanations(conn, explanations_by_layer, model_id, source_set_name, user_id)
        
        # Insert activations if available
        total_activations_inserted = 0
        if upload_activations_too and activations_by_layer:
            print("\n🎯 Inserting activations...")
            total_activations_inserted = upload_activations(conn, activations_by_layer, model_id, source_set_name, user_id)
        
        print(f"\n🎉 Upload complete!")
        print(f"Total explanations inserted: {total_explanations_inserted}")
        if total_activations_inserted > 0:
            print(f"Total activations inserted: {total_activations_inserted}")
        print(f"\n📊 View your features at:")
        print(f"   http://localhost:3000/{model_id}")
        
        if explanations_by_layer:
            example_layer = min(explanations_by_layer.keys())
            print(f"   Example: http://localhost:3000/{model_id}/{example_layer}-{source_set_name}")
    
    finally:
        conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_explanations_complete.py <explanations_dir> [source_set_name]")
        print("Example: python upload_explanations_complete.py ~/Desktop/code/20-realm-l20r-8x/explanations")
        sys.exit(1)
    
    explanations_dir = sys.argv[1]
    source_set_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    upload_explanations_complete(explanations_dir, source_set_name)

if __name__ == "__main__":
    main()
