# load_model.py
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from subfield_gnn import SubfieldGNN
from interdisciplinary_report import get_interdisciplinary_growth_report, visualize_top_interdisciplinary_collaborations

def load_saved_model(model_path, data_path):
    """
    Load a previously saved GNN model and run predictions.
    
    Args:
        model_path: Path to the saved model file
        data_path: Path to the original data file used to train the model
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        return
        
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found")
        return
    
    print(f"Loading model from '{model_path}'...")
    try:
        # Load model
        model = torch.load(model_path)
        print("Model loaded successfully")
        
        # Initialize data handler with the same data
        print(f"Loading data from '{data_path}'...")
        subfield_gnn = SubfieldGNN(data_path)
        
        # Check if we have data
        months = sorted(subfield_gnn.monthly_graphs.keys())
        print(f"Found data for {len(months)} months: {months}")
        
        # Prepare data with the same window size as before
        window_size = 6  # Use the same as in training
        prediction_horizon = 6
        print("Preparing temporal data...")
        subfield_gnn.prepare_temporal_data(window_size, prediction_horizon)
        
        # Generate predictions and reports
        print("Generating predictions and reports...")
        
        # Generate interdisciplinary growth report with top_k=40
        growth_df = get_interdisciplinary_growth_report(model, subfield_gnn, top_k=40)
        
        # Visualize if we have results
        if not growth_df.empty:
            visualize_top_interdisciplinary_collaborations(growth_df, subfield_gnn)
            print("\nAnalysis complete. Check output files for results.")
        else:
            print("No interdisciplinary collaborations found in the data")
        
        return model, subfield_gnn
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a saved subfield collaboration GNN model')
    parser.add_argument('--model', default="subfield_gnn_model.pt", 
                        help='Path to the saved model file')
    parser.add_argument('--data', default="../data/all_articles_2023_2024.jsonl", 
                        help='Path to the data file')
    args = parser.parse_args()
    
    print("Starting analysis using saved model...")
    print("This will analyze collaborations with > 3 existing partnerships")
    model, subfield_gnn = load_saved_model(args.model, args.data)