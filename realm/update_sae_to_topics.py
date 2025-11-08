#!/usr/bin/env python3
"""
Update existing sae_to_topics.json with new explanations from batch-autointerp output.

This script reads an existing sae_to_topics.json file and updates only:
- sae_label (from the first explanation)
- activation_density (from feature data)
- explanations array in the "other" section

All other fields (in_tokens, out_tokens, entropies, etc.) are preserved.
"""

import json
import gzip
import glob
import os
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


def process_batch_files(data_dir: str, max_features: Optional[int] = None) -> Dict[str, Any]:
    """Process batch files and combine data by feature index, stopping early if max_features is set."""

    print(f"Processing data from: {data_dir}")

    # Collect all data by feature index
    features_by_index = {}
    explanations_by_index = defaultdict(list)

    # Get sorted file lists
    feature_files = sorted(glob.glob(os.path.join(data_dir, "features", "batch-*.jsonl.gz")))
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
        'explanations': explanations_by_index
    }


def update_sae_to_topics(data_dir: str, existing_file: str, output_file: str = None,
                        max_features: Optional[int] = None) -> None:
    """Update existing sae_to_topics.json file with new explanations from batch-autointerp data."""

    # Load existing sae_to_topics.json
    print(f"Loading existing file: {existing_file}")
    with open(existing_file, 'r') as f:
        sae_to_topics = json.load(f)

    print(f"Loaded {len(sae_to_topics)} existing features")

    # Load all data from batch files
    data = process_batch_files(data_dir, max_features)
    features = data['features']
    explanations_by_index = data['explanations']

    # Update the sae_to_topics structure
    print("Updating sae_to_topics.json structure...")

    updated_count = 0
    skipped_count = 0

    feature_indices = sorted(features.keys(), key=int)
    if max_features:
        feature_indices = feature_indices[:max_features]

    for idx in feature_indices:
        if int(idx) % 1000 == 0:
            print(f"  Processing feature {idx}...")

        # Skip if this feature doesn't exist in the original file
        if idx not in sae_to_topics:
            skipped_count += 1
            continue

        feature = features[idx]
        explanations = explanations_by_index.get(idx, [])

        # Only update if we have explanations for this feature
        if not explanations:
            continue

        # Get primary explanation description
        sae_label = explanations[0].get('description', 'unknown')

        # Calculate activation density (frac_nonzero from feature data)
        activation_density = feature.get('frac_nonzero', 0.0)

        # Format explanations for the "other" section
        formatted_explanations = []
        for exp in explanations:
            formatted_explanations.append({
                "typeName": exp.get('typeName', 'unknown'),
                "explanationModelName": exp.get('explanationModelName', 'unknown'),
                "description": exp.get('description', '')
            })

        # Update only the specific fields, preserving everything else
        sae_to_topics[idx]["sae_label"] = sae_label
        sae_to_topics[idx]["activation_density"] = activation_density

        # Ensure "other" section exists
        if "other" not in sae_to_topics[idx]:
            sae_to_topics[idx]["other"] = {}

        # Add explanations to the "other" section
        sae_to_topics[idx]["other"]["explanations"] = formatted_explanations

        updated_count += 1

    print(f"\nUpdated {updated_count} features")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} features (not in existing file)")

    # Determine output file path
    if output_file is None:
        output_file = existing_file

    # Write the updated file
    print(f"Writing {len(sae_to_topics)} features to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(sae_to_topics, f, indent=2)

    print(f"Successfully updated {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Update existing sae_to_topics.json with batch-autointerp data')
    parser.add_argument('data_dir', help='Directory containing explanations/ and features/ subdirectories')
    parser.add_argument('--existing-file', required=True, help='Path to existing sae_to_topics.json file to update')
    parser.add_argument('-o', '--output', help='Output JSON file (default: overwrites existing file)')
    parser.add_argument('--max-features', type=int, help='Maximum number of features to process (for testing)')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return 1

    if not os.path.exists(args.existing_file):
        print(f"Error: Existing file {args.existing_file} does not exist")
        return 1

    # Check for required subdirectories
    required_dirs = ['explanations', 'features']
    for req_dir in required_dirs:
        full_path = os.path.join(args.data_dir, req_dir)
        if not os.path.exists(full_path):
            print(f"Error: Required subdirectory {full_path} does not exist")
            return 1

    try:
        update_sae_to_topics(args.data_dir, args.existing_file, args.output, args.max_features)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
