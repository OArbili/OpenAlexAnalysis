# run_model.py
import os
from subfield_gnn import SubfieldGNN, train_collaboration_predictor_with_sliding_windows, predict_future_collaborations, set_seed
from interdisciplinary_report import generate_interdisciplinary_report, visualize_top_interdisciplinary_collaborations, get_interdisciplinary_growth_report
import argparse
import time
import torch

# Path to your data file - this can be overridden by debug_run.py
data_path = "../data/all_articles_2023_2024.jsonl"  # Your uploaded data file

# Define SimpleGNN outside of main() function so it can be serialized
class SimpleGNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.ones(1))
        
    def forward(self, data):
        # Just return current values slightly increased (dummy prediction)
        return data.y * 1.1

def main(debug=False):
    start_time = time.time()
    print(f"Running in {'DEBUG' if debug else 'FULL'} mode")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create model save path
    model_save_path = "subfield_gnn_model.pt"
    if debug:
        model_save_path = "subfield_gnn_model_debug.pt"
    
    print("Initializing SubfieldGNN...")
    # Initialize data handler
    subfield_gnn = SubfieldGNN(data_path)
    
    # Check if we have enough data
    months = sorted(subfield_gnn.monthly_graphs.keys())
    print(f"Found data for {len(months)} months: {months}")
    
    if len(months) < 2:
        print("WARNING: Not enough monthly data. Need at least 2 months.")
        if not debug:
            print("Try running with --debug flag to continue anyway.")
            return
    
    # Configure window size and stride for article-based sliding windows
    # In debug mode, use smaller window size and fewer epochs
    if debug:
        window_size = 1000  # Use 1000 articles per window in debug mode
        stride = 500        # Slide by 500 articles
        epochs = 5          # Use very few epochs in debug mode
        print(f"DEBUG MODE: Using window_size={window_size}, stride={stride}, epochs={epochs}")
    else:
        total_articles = len(subfield_gnn.articles)
        # Use approximately 10% of articles per window, with 5% stride
        window_size = max(5000, int(total_articles * 0.1))
        stride = max(2500, int(window_size * 0.5))
        epochs = 100
        print(f"FULL MODE: Using window_size={window_size} (~{window_size/total_articles:.1%} of data), "
              f"stride={stride}, epochs={epochs}")
    
    # Train the model with sliding window approach
    print("Training model with sliding windows of articles...")
    training_start = time.time()
    
    # Use a custom simple model in debug mode
    if debug:
        # Create dummy model (now defined outside main())
        model = SimpleGNN()
        print("DEBUG MODE: Using simplified model that skips actual training")
        
        # Create at least some training data for consistency
        subfield_gnn.create_sliding_window_graphs(window_size, stride)
        
        # Save the dummy model for consistency
        torch.save(model, model_save_path)
        print(f"DEBUG MODEL saved to '{model_save_path}'")
    else:
        # Train the actual model with sliding windows
        model, subfield_gnn = train_collaboration_predictor_with_sliding_windows(
            data_path, 
            window_size=window_size,
            stride=stride,
            epochs=epochs,
            model_save_path=model_save_path
        )
        
        # Verify the model was saved
        if os.path.exists(model_save_path):
            print(f"Verified model was saved at '{model_save_path}'")
        else:
            print(f"WARNING: Model save verification failed. Saving model now.")
            torch.save(model, model_save_path)
    
    training_time = time.time() - training_start
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Generate report
    print("Generating predictions and reports...")
    try:
        # Generate interdisciplinary growth report with top_k=40
        growth_df = generate_interdisciplinary_report(model, subfield_gnn, top_k=40, debug=debug)
        
        # Visualize the interdisciplinary network if we have results
        if growth_df is not None and not growth_df.empty:
            visualize_top_interdisciplinary_collaborations(growth_df, subfield_gnn, debug=debug)
        else:
            print("No interdisciplinary collaborations found in the data")
    except Exception as e:
        print(f"Error generating reports: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"Pipeline completed in {total_time:.2f} seconds")
    print(f"Done! Model saved at '{model_save_path}'. Check the current directory for report files and visualizations.")
    print("To analyze the model later, use load_model.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run subfield collaboration GNN model')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode with minimal computation')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Number of articles per sliding window (defaults to 10% of total articles)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride between sliding windows (defaults to half of window size)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed from command line
    set_seed(args.seed)
    
    # Override defaults if provided via command line
    if args.window_size:
        window_size = args.window_size
    if args.stride:
        stride = args.stride
    if args.epochs:
        epochs = args.epochs
    
    main(debug=args.debug)