import os
import sys
from subfield_gnn import SubfieldGNN
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import traceback

from interdisciplinary_report import get_interdisciplinary_growth_report, generate_interdisciplinary_report, visualize_top_interdisciplinary_collaborations


def run_modified_analysis(model_path="subfield_gnn_model.pt", 
                         data_path="../data/all_articles_2023_2024.jsonl",
                         top_k=30,
                         min_collaborations=1):
    """
    Run analysis with our modified functions to generate more comprehensive interdisciplinary reports
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the data file
        top_k: Number of top collaborations to include
        min_collaborations: Minimum number of current collaborations to consider
    """
    start_time = time.time()
    print(f"Running modified analysis with min_collaborations={min_collaborations} and top_k={top_k}")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Check if the data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    try:
        # Initialize data handler
        print("Initializing SubfieldGNN...")
        subfield_gnn = SubfieldGNN(data_path)
        
        # Check if we have enough data
        months = sorted(subfield_gnn.monthly_graphs.keys())
        print(f"Found data for {len(months)} months: {months}")
        
        if len(months) < 2:
            print("WARNING: Not enough monthly data. Need at least 2 months.")
            return
        
        # Prepare temporal data
        window_size = min(6, len(months) - 1)
        prediction_horizon = min(6, len(months) - window_size)
        
        print("Preparing temporal data...")
        subfield_gnn.prepare_temporal_data(window_size, prediction_horizon)
        
        print(f"Number of training examples: {len(subfield_gnn.training_data)}")
        if len(subfield_gnn.training_data) == 0:
            print("ERROR: No training examples could be created. Check your data.")
            return
        
        # Load the trained model
        print(f"Loading model from {model_path}...")
        model = torch.load(model_path)
        print("Model loaded successfully")
        
        # Generate report with modified parameters
        print(f"Generating predictions and reports with top_k={top_k}, min_collaborations={min_collaborations}...")
        
        # Generate interdisciplinary report with our modified function
        growth_df = generate_interdisciplinary_report(model, subfield_gnn, top_k=top_k)
        
        if growth_df is not None and not growth_df.empty:
            print(f"Generated report with {len(growth_df)} collaborations")
            
            # Generate the network visualization with our modified function
            visualize_top_interdisciplinary_collaborations(growth_df, subfield_gnn)
        else:
            print("No interdisciplinary collaborations found in the data")
        
        total_time = time.time() - start_time
        print(f"Modified analysis completed in {total_time:.2f} seconds")
        print("Done! Check the current directory for report files and visualizations.")
        
        return growth_df
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run modified subfield collaboration analysis')
    parser.add_argument('--model', default="subfield_gnn_model.pt", 
                        help='Path to the trained model file')
    parser.add_argument('--data', default="../data/all_articles_2023_2024.jsonl", 
                        help='Path to the data file')
    parser.add_argument('--top_k', type=int, default=50, 
                        help='Number of top collaborations to include')
    parser.add_argument('--min_collaborations', type=int, default=1, 
                        help='Minimum number of current collaborations to consider')
    
    args = parser.parse_args()
    
    run_modified_analysis(
        model_path=args.model,
        data_path=args.data,
        top_k=args.top_k,
        min_collaborations=args.min_collaborations
    )