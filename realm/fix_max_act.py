#!/usr/bin/env python3
"""
Fix maxActApprox values for features that have activations but maxActApprox = 0.
This solves the issue where features show "This feature has no known activations" 
even though they have valid activation data.

Usage: python fix_max_act_approx.py [model_id] [layer_pattern]
"""

import sys
import psycopg2
from typing import Optional

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

def fix_max_act_approx(model_id: Optional[str] = None, layer_pattern: Optional[str] = None):
    """Fix maxActApprox values for features with activations."""
    
    conn = connect_to_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    try:
        # Build WHERE clause based on parameters
        where_conditions = ['n."maxActApprox" = 0 OR n."maxActApprox" IS NULL']
        params = []
        
        if model_id:
            where_conditions.append('n."modelId" = %s')
            params.append(model_id)
        
        if layer_pattern:
            where_conditions.append('n.layer LIKE %s')
            params.append(f'%{layer_pattern}%')
        
        where_clause = ' AND '.join(where_conditions)
        
        # Query to find features with maxActApprox = 0 but have activations
        query = f'''
            SELECT 
                n."modelId", 
                n.layer, 
                n.index, 
                n."maxActApprox",
                MAX(a."maxValue") as actual_max_value,
                COUNT(a.id) as activation_count
            FROM "Neuron" n
            LEFT JOIN "Activation" a ON (
                n."modelId" = a."modelId" AND 
                n.layer = a.layer AND 
                n.index = a.index
            )
            WHERE {where_clause}
            GROUP BY n."modelId", n.layer, n.index, n."maxActApprox"
            HAVING MAX(a."maxValue") IS NOT NULL
            ORDER BY n."modelId", n.layer, CAST(n.index AS INTEGER)
        '''
        
        print(f"Finding features with maxActApprox = 0 but have activations...")
        if model_id:
            print(f"  Model filter: {model_id}")
        if layer_pattern:
            print(f"  Layer filter: *{layer_pattern}*")
        
        cursor.execute(query, params)
        features = cursor.fetchall()
        
        print(f"Found {len(features)} features to fix")
        
        if not features:
            print("No features need fixing!")
            return
        
        # Show some examples
        print("\nExamples of features to fix:")
        for i, (model_id, layer, index, current_max, actual_max, act_count) in enumerate(features[:5]):
            print(f"  {model_id}/{layer}/{index}: {current_max} -> {actual_max:.3f} ({act_count} activations)")
        
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")
        
        # Confirm update
        response = input(f"\nUpdate maxActApprox for {len(features)} features? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return
        
        # Update features
        print("\nUpdating features...")
        updated_count = 0
        
        for model_id, layer, index, current_max, actual_max, act_count in features:
            try:
                cursor.execute('''
                    UPDATE "Neuron" 
                    SET "maxActApprox" = %s 
                    WHERE "modelId" = %s AND layer = %s AND index = %s
                ''', (actual_max, model_id, layer, index))
                
                if cursor.rowcount > 0:
                    updated_count += 1
                    if updated_count % 100 == 0:
                        print(f"  Updated {updated_count} features...")
                        
            except Exception as e:
                print(f"Error updating {model_id}/{layer}/{index}: {e}")
                conn.rollback()
                return
        
        # Commit all changes
        conn.commit()
        print(f"\n✅ Successfully updated {updated_count} features!")
        print("\nNow your features should display properly in the webapp.")
        
    except Exception as e:
        print(f"Error fixing maxActApprox: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def main():
    model_id = None
    layer_pattern = None
    
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    if len(sys.argv) > 2:
        layer_pattern = sys.argv[2]
    
    print("Fix maxActApprox values for features")
    print("=====================================")
    
    fix_max_act_approx(model_id, layer_pattern)

if __name__ == "__main__":
    main()