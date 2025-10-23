#!/usr/bin/env python3
"""
Generate sae_to_topics.json from batch-autointerp output files.

This script processes activation, explanation, and feature files to generate
a JSON structure compatible with SAE topic analysis tools.
"""

import json
import gzip
import glob
import os
import math
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse


def load_jsonl_gz(file_path: str) -> List[Dict]:
    """Load data from a gzipped JSONL file."""
    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_entropy(token_probs: List[List]) -> float:
    """Calculate Shannon entropy from token probabilities."""
    if not token_probs:
        return 0.0
    
    total_prob = sum(prob for _, _, prob in token_probs)
    if total_prob == 0:
        return 0.0
    
    entropy = 0.0
    for _, _, prob in token_probs:
        if prob > 0:
            normalized_prob = prob / total_prob
            entropy -= normalized_prob * math.log2(normalized_prob)
    
    return entropy


def get_top_tokens_with_probs(feature_data: Dict, pos_str: List[str], pos_values: List[float], 
                             neg_str: List[str], neg_values: List[float], top_k: int = 20) -> tuple:
    """Extract top tokens with their probabilities for in_tokens and out_tokens."""
    
    # For in_tokens (positive activating tokens)
    in_tokens = []
    if pos_str and pos_values:
        for i, (token, value) in enumerate(zip(pos_str[:top_k], pos_values[:top_k])):
            # Use token index as a placeholder - in real scenario you'd need token vocab
            token_id = 1000 + i  # Placeholder token ID
            in_tokens.append([token, token_id, float(value)])
    
    # For out_tokens (negative/inhibiting tokens) 
    out_tokens = []
    if neg_str and neg_values:
        for i, (token, value) in enumerate(zip(neg_str[:top_k], neg_values[:top_k])):
            # Use token index as a placeholder
            token_id = 2000 + i  # Placeholder token ID  
            out_tokens.append([token, token_id, float(value)])
    
    return in_tokens, out_tokens


def process_batch_files(data_dir: str, max_features: Optional[int] = None) -> Dict[str, Any]:
    """Process batch files and combine data by feature index, stopping early if max_features is set."""
    
    print(f"Processing data from: {data_dir}")
    
    # Collect all data by feature index
    features_by_index = {}
    activations_by_index = defaultdict(list)
    explanations_by_index = defaultdict(list)
    
    # Get sorted file lists
    feature_files = sorted(glob.glob(os.path.join(data_dir, "features", "batch-*.jsonl.gz")))
    activation_files = sorted(glob.glob(os.path.join(data_dir, "activations", "batch-*.jsonl.gz")))
    explanation_files = sorted(glob.glob(os.path.join(data_dir, "explanations", "batch-*.jsonl.gz")))
    
    target_features = max_features or float('inf')
    
    # Process feature files first (stop early if we have enough)
    print("Loading feature files...")
    for feature_file in feature_files:
        print(f"  Processing {os.path.basename(feature_file)}")
        features = load_jsonl_gz(feature_file)
        for feature in features:
            idx = str(feature['index'])
            features_by_index[idx] = feature
            
            # Stop early if we have enough features
            if len(features_by_index) >= target_features:
                print(f"  Reached target of {target_features} features, stopping early")
                break
        
        if len(features_by_index) >= target_features:
            break
    
    print(f"Loaded {len(features_by_index)} features")
    
    # Get the indices we actually need
    target_indices = set(features_by_index.keys())
    
    # Process activation files (only for features we have)
    print("Loading activation files...")
    for activation_file in activation_files:
        print(f"  Processing {os.path.basename(activation_file)}")
        activations = load_jsonl_gz(activation_file)
        for activation in activations:
            idx = str(activation['index'])
            if idx in target_indices:
                activations_by_index[idx].append(activation)
    
    print(f"Loaded activations for {len(activations_by_index)} features")
    
    # Process explanation files (only for features we have)
    print("Loading explanation files...")
    for explanation_file in explanation_files:
        print(f"  Processing {os.path.basename(explanation_file)}")
        explanations = load_jsonl_gz(explanation_file)
        for explanation in explanations:
            idx = str(explanation['index'])
            if idx in target_indices:
                explanations_by_index[idx].append(explanation)
    
    print(f"Loaded explanations for {len(explanations_by_index)} features")
    
    return {
        'features': features_by_index,
        'activations': activations_by_index,
        'explanations': explanations_by_index
    }


def generate_sae_to_topics(data_dir: str, output_file: str = "sae_to_topics.json", 
                          max_features: Optional[int] = None) -> None:
    """Generate the sae_to_topics.json file from batch-autointerp data."""
    
    # Load all data
    data = process_batch_files(data_dir, max_features)
    features = data['features']
    activations_by_index = data['activations']
    explanations_by_index = data['explanations']
    
    # Generate the sae_to_topics structure
    print("Generating sae_to_topics.json structure...")
    sae_to_topics = {}
    
    feature_indices = sorted(features.keys(), key=int)
    if max_features:
        feature_indices = feature_indices[:max_features]
    
    for idx in feature_indices:
        if int(idx) % 1000 == 0:
            print(f"  Processing feature {idx}...")
            
        feature = features[idx]
        activations = activations_by_index.get(idx, [])
        explanations = explanations_by_index.get(idx, [])
        
        # Get primary explanation description
        sae_label = "unknown"
        if explanations:
            sae_label = explanations[0].get('description', 'unknown')
        
        # Calculate activation density (frac_nonzero from feature data)
        activation_density = feature.get('frac_nonzero', 0.0)
        
        # Get top tokens from feature data
        pos_str = feature.get('pos_str', [])
        pos_values = feature.get('pos_values', [])
        neg_str = feature.get('neg_str', [])
        neg_values = feature.get('neg_values', [])
        
        in_tokens, out_tokens = get_top_tokens_with_probs(
            feature, pos_str, pos_values, neg_str, neg_values
        )
        
        # Calculate entropies
        in_entropy = calculate_entropy(in_tokens)
        out_entropy = calculate_entropy(out_tokens)
        
        # Format explanations for the "other" section
        formatted_explanations = []
        for exp in explanations:
            formatted_explanations.append({
                "typeName": exp.get('typeName', 'unknown'),
                "explanationModelName": exp.get('explanationModelName', 'unknown'),  
                "description": exp.get('description', '')
            })
        
        # Build the feature entry
        feature_entry = {
            "sae_label": sae_label,
            "sub_topic_label": "None",
            "topic_label": "None", 
            "csp": False,
            "detection": "NONE",
            "threshold": 0.0,
            "verified": 0,
            "display": 0,
            "other": {
                "out_tokens": out_tokens,
                "out_entropy": out_entropy,
                "in_tokens": in_tokens,
                "in_entropy": in_entropy,
                "explanations": formatted_explanations
            },
            "activation_density": activation_density
        }
        
        sae_to_topics[idx] = feature_entry
    
    # Write the output file
    print(f"Writing {len(sae_to_topics)} features to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sae_to_topics, f, indent=2)
    
    print(f"Successfully generated {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate sae_to_topics.json from batch-autointerp data')
    parser.add_argument('data_dir', help='Directory containing activations/, explanations/, and features/ subdirectories')
    parser.add_argument('-o', '--output', default='sae_to_topics.json', help='Output JSON file (default: sae_to_topics.json)')
    parser.add_argument('--max-features', type=int, help='Maximum number of features to process (for testing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return 1
    
    # Check for required subdirectories
    required_dirs = ['activations', 'explanations', 'features']
    for req_dir in required_dirs:
        full_path = os.path.join(args.data_dir, req_dir)
        if not os.path.exists(full_path):
            print(f"Error: Required subdirectory {full_path} does not exist")
            return 1
    
    try:
        generate_sae_to_topics(args.data_dir, args.output, args.max_features)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())