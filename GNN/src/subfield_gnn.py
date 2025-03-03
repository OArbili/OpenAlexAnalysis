import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import torch
import os
import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def predict_future_collaborations(model, subfield_gnn, top_k=10):
    """
    Predict future collaborations between subfields for the next period
    after the validation period.
    
    Args:
        model (TemporalGNN): Trained GNN model
        subfield_gnn (SubfieldGNN): Data handler
        top_k (int): Number of top predictions to return
    
    Returns:
        pandas.DataFrame: DataFrame with predictions
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get the latest data for prediction
    latest_data = subfield_gnn.training_data[-1].to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(latest_data).cpu().numpy()
    
    # Create prediction results
    results = []
    for i, edge_idx in enumerate(latest_data.edge_index.t().cpu().numpy()):
        if i % 2 == 0:  # Process each undirected edge once
            src, dst = edge_idx
            src_id = subfield_gnn.reverse_mapping[src]
            dst_id = subfield_gnn.reverse_mapping[dst]
            
            src_name = subfield_gnn.subfield_names[src_id]
            dst_name = subfield_gnn.subfield_names[dst_id]
            
            prediction = predictions[i]
            
            results.append({
                'source_id': src_id,
                'source_name': src_name,
                'target_id': dst_id,
                'target_name': dst_name,
                'predicted_collaborations': prediction,
                'current_collaborations': latest_data.y[i].item()
            })
    
    # Convert to DataFrame and sort by predicted collaborations
    results_df = pd.DataFrame(results)
    results_df['predicted_growth'] = results_df['predicted_collaborations'] - results_df['current_collaborations']
    results_df['predicted_growth_percent'] = ((results_df['predicted_collaborations'] / 
                                              results_df['current_collaborations']) - 1) * 100
    
    # Handle division by zero
    results_df.loc[results_df['current_collaborations'] == 0, 'predicted_growth_percent'] = np.inf
    
    # Sort by predicted growth
    results_df = results_df.sort_values('predicted_growth', ascending=False)
    
    return results_df.head(top_k)
    
def train_collaboration_predictor_with_sliding_windows(data_path, window_size=10000, stride=5000, 
                                                      epochs=100, model_save_path="subfield_gnn_model.pt"):
    """
    Train a GNN model to predict subfield collaborations using sliding windows of articles.
    
    Args:
        data_path (str): Path to the data file
        window_size (int): Number of articles in each window
        stride (int): Number of articles to shift for the next window
        epochs (int): Number of training epochs
        model_save_path (str): Path to save the trained model
    
    Returns:
        tuple: Trained model and subfield mapping
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize and prepare data
    subfield_gnn = SubfieldGNN(data_path)
    
    # Create sliding window graph samples instead of monthly ones
    graph_samples = subfield_gnn.create_sliding_window_graphs(window_size, stride)
    
    if len(graph_samples) == 0:
        print("Error: No graph samples could be created.")
        return None, subfield_gnn
    
    # Split data chronologically: first 75% for training, remainder for validation
    split_idx = int(len(graph_samples) * 0.75)
    train_data = graph_samples[:split_idx]
    validation_data = graph_samples[split_idx:]
    
    print(f"Training on {len(train_data)} samples, validating on {len(validation_data)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dimensions from training data
    if len(train_data) > 0:
        input_node_dim = train_data[0].x.shape[1]
        input_edge_dim = train_data[0].edge_attr.shape[1]
    else:
        # Fallback if training data is empty
        print("Warning: Empty training data. Using dimensions from all available data.")
        input_node_dim = subfield_gnn.training_data[0].x.shape[1]
        input_edge_dim = subfield_gnn.training_data[0].edge_attr.shape[1]
    
    model = TemporalGNN(input_node_dim, input_edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Track best validation loss for model saving
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 10  # For early stopping
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch)
            loss = criterion(pred, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss/len(train_loader) if train_loader.dataset else float('inf')
        
        # Evaluation on validation set
        if epoch % 5 == 0 or epoch == epochs-1:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in validation_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    val_loss += criterion(pred, batch.y).item()
            
            avg_val_loss = val_loss/len(validation_loader) if validation_loader.dataset else float('inf')
            
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save best model and implement early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save the best model
                torch.save(model, model_save_path)
                print(f"New best model saved at epoch {epoch} with validation loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}, best was epoch {best_epoch}")
                    break
    
    # Load the best model for return
    if os.path.exists(model_save_path):
        print(f"Loading best model from epoch {best_epoch}")
        model = torch.load(model_save_path)
    
    return model, subfield_gnn


class SubfieldGNN:
    def __init__(self, data_path):
        """
        Initialize the GNN model for predicting subfield collaborations.
        
        Args:
            data_path (str): Path to the data file containing articles with topics and subfields
        """
        self.data_path = data_path
        self.subfield_mapping = {}  # Maps subfield IDs to indices
        self.reverse_mapping = {}   # Maps indices to subfield IDs
        self.subfield_names = {}    # Maps subfield IDs to display names
        self.monthly_graphs = {}    # Stores monthly collaboration graphs
        self.collaboration_history = {}  # Stores historical collaboration data
        self.training_data = []     # Training data for the GNN
        
        # Load and process data
        self.load_data()
        self.process_data()
    
    def load_data(self):
        """Load and parse the JSON data from the file."""
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        
        self.articles = []
        for line in lines:
            if line.strip():
                self.articles.append(json.loads(line))
        
        print(f"Loaded {len(self.articles)} articles")
    
    def process_data(self):
        """
        Process the data to extract subfields and create monthly graphs.
        For a 2-year dataset, we expect to see 24 months of data that will be
        split into 18 months for training and 6 months for validation.
        """
        # Extract all unique subfields
        unique_subfields = set()
        for article in self.articles:
            for topic in article.get('topics', []):
                if 'subfield' in topic and topic['subfield']:
                    subfield_id = topic['subfield']['id']
                    unique_subfields.add(subfield_id)
                    self.subfield_names[subfield_id] = topic['subfield']['display_name']
        
        # Create mapping from subfield IDs to indices
        for idx, subfield_id in enumerate(sorted(unique_subfields)):
            self.subfield_mapping[subfield_id] = idx
            self.reverse_mapping[idx] = subfield_id
        
        print(f"Found {len(unique_subfields)} unique subfields")
        
        # Group articles by month
        articles_by_month = {}
        for article in self.articles:
            pub_date = article.get('publication_date')
            if pub_date:
                date_obj = datetime.strptime(pub_date, "%Y-%m-%d")
                month_key = f"{date_obj.year}-{date_obj.month:02d}"
                
                if month_key not in articles_by_month:
                    articles_by_month[month_key] = []
                
                articles_by_month[month_key].append(article)
        
        # Create monthly collaboration graphs
        for month, month_articles in articles_by_month.items():
            # Initialize adjacency matrix for this month
            n_subfields = len(self.subfield_mapping)
            adj_matrix = np.zeros((n_subfields, n_subfields))
            
            # Build the adjacency matrix based on co-occurrences in articles
            for article in month_articles:
                article_subfields = []
                
                for topic in article.get('topics', []):
                    if 'subfield' in topic and topic['subfield']:
                        subfield_id = topic['subfield']['id']
                        article_subfields.append(self.subfield_mapping[subfield_id])
                
                # Create edges between all pairs of subfields in this article
                for i, sf1 in enumerate(article_subfields):
                    for sf2 in article_subfields[i:]:
                        adj_matrix[sf1, sf2] += 1
                        adj_matrix[sf2, sf1] += 1
            
            self.monthly_graphs[month] = adj_matrix
            
            # Also store the collaboration data for features
            for i in range(n_subfields):
                for j in range(i+1, n_subfields):
                    if adj_matrix[i, j] > 0:
                        collab_key = f"{self.reverse_mapping[i]}--{self.reverse_mapping[j]}"
                        
                        if collab_key not in self.collaboration_history:
                            self.collaboration_history[collab_key] = {}
                        
                        self.collaboration_history[collab_key][month] = adj_matrix[i, j]
        
        print(f"Created graphs for {len(self.monthly_graphs)} months")
    
    def create_sliding_window_graphs(self, window_size=10000, stride=5000):
        """
        Create graphs from sliding windows of articles instead of monthly buckets.
        
        Args:
            window_size: Number of articles in each window
            stride: Number of articles to shift for the next window
        
        Returns:
            List of graph samples for training
        """
        print(f"Creating sliding window graphs with window_size={window_size}, stride={stride}")
        
        # Sort articles by date
        sorted_articles = sorted(self.articles, 
                               key=lambda x: x.get('publication_date', ''))
        
        # Skip articles without publication dates
        sorted_articles = [a for a in sorted_articles if a.get('publication_date', '')]
        
        if not sorted_articles:
            print("No articles with valid publication dates found!")
            return []
        
        # Initialize training data list
        self.training_data = []
        
        # Create windows of articles
        num_windows = max(1, (len(sorted_articles) - window_size) // stride + 1)
        print(f"Creating {num_windows} sliding windows from {len(sorted_articles)} articles")
        
        for i in range(0, len(sorted_articles) - window_size, stride):
            # Get current window and next window for prediction
            current_window = sorted_articles[i:i+window_size]
            next_window_idx = min(i+window_size+stride, len(sorted_articles)-1)
            next_window = sorted_articles[i+window_size:next_window_idx]
            
            if not next_window:  # Skip if no next window (end of data)
                continue
                
            # Create graph from current window
            current_graph = self._create_graph_from_articles(current_window)
            
            # Create graph from next window (target for prediction)
            next_graph = self._create_graph_from_articles(next_window)
            
            # Create a PyG Data object
            graph_data = self._convert_to_pyg_data(current_graph, next_graph)
            
            if graph_data is not None:
                self.training_data.append(graph_data)
        
        print(f"Created {len(self.training_data)} graph samples for training")
        return self.training_data

    def _create_graph_from_articles(self, articles):
        """Create a graph from a list of articles"""
        n_subfields = len(self.subfield_mapping)
        adj_matrix = np.zeros((n_subfields, n_subfields))
        
        # Track subfield features
        node_features = np.zeros((n_subfields, 3))  # [degree, article_count, recency]
        
        # Track when each subfield was last seen (for recency features)
        latest_dates = {}
        
        for article in articles:
            pub_date = datetime.strptime(article.get('publication_date'), "%Y-%m-%d")
            article_subfields = []
            
            for topic in article.get('topics', []):
                if 'subfield' in topic and topic['subfield']:
                    subfield_id = topic['subfield']['id']
                    if subfield_id in self.subfield_mapping:
                        sf_idx = self.subfield_mapping[subfield_id]
                        article_subfields.append(sf_idx)
                        
                        # Update node features
                        node_features[sf_idx, 1] += 1  # Increment article count
                        
                        # Update latest date
                        if sf_idx not in latest_dates or pub_date > latest_dates[sf_idx]:
                            latest_dates[sf_idx] = pub_date
            
            # Create edges between all pairs of subfields in this article
            for i, sf1 in enumerate(article_subfields):
                for sf2 in article_subfields[i+1:]:
                    adj_matrix[sf1, sf2] += 1
                    adj_matrix[sf2, sf1] += 1
        
        # Calculate degree for each node (total collaborations)
        for i in range(n_subfields):
            node_features[i, 0] = np.sum(adj_matrix[i])
        
        # Calculate recency feature (days from most recent to last article)
        if articles:
            end_date = datetime.strptime(articles[-1].get('publication_date'), "%Y-%m-%d")
            for sf_idx, latest_date in latest_dates.items():
                # Recency as days from latest appearance to end of window, normalized
                days_diff = (end_date - latest_date).days
                node_features[sf_idx, 2] = np.exp(-days_diff / 30)  # Exponential decay over 30 days
        
        return adj_matrix, node_features

    def _convert_to_pyg_data(self, current_data, next_data):
        """Convert adjacency matrices to PyG Data objects"""
        current_adj, current_features = current_data
        next_adj, _ = next_data
        
        # Get edges with non-zero weights
        edge_index = []
        edge_attr = []
        labels = []
        
        n_subfields = current_adj.shape[0]
        
        for src in range(n_subfields):
            for dst in range(src+1, n_subfields):
                # Current collaboration strength
                current_weight = current_adj[src, dst]
                
                # Target collaboration strength (from next window)
                next_weight = next_adj[src, dst]
                
                # Only include edges that have some collaboration in either window
                if current_weight > 0 or next_weight > 0:
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])  # Undirected graph
                    
                    # Edge feature is just the current weight
                    edge_attr.append([current_weight])
                    edge_attr.append([current_weight])  # Same for both directions
                    
                    # Label is the collaboration strength in next window
                    labels.append(next_weight)
                    labels.append(next_weight)  # Same for both directions
        
        if not edge_index:  # Skip if no edges
            return None
            
        # Convert to PyTorch tensors
        x = torch.FloatTensor(current_features)
        edge_index = torch.LongTensor(edge_index).t().contiguous()  # PyG expects shape [2, num_edges]
        edge_attr = torch.FloatTensor(edge_attr)
        y = torch.FloatTensor(labels)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    
    def prepare_temporal_data(self, window_size=6, prediction_horizon=6):
        """
        Prepare temporal data for training the GNN.
        Data will be chronologically ordered so we can later split into:
        - First 18 months for training
        - Months 19-24 for validation
        
        Args:
            window_size (int): Number of months to use as input features
            prediction_horizon (int): Number of months ahead to predict
        """
        # Sort months chronologically
        sorted_months = sorted(self.monthly_graphs.keys())
        
        if len(sorted_months) < window_size + prediction_horizon:
            raise ValueError(f"Not enough data. Need at least {window_size + prediction_horizon} months.")
        
        n_subfields = len(self.subfield_mapping)
        
        # For each valid time window
        for i in range(len(sorted_months) - window_size - prediction_horizon + 1):
            input_months = sorted_months[i:i+window_size]
            target_month = sorted_months[i+window_size+prediction_horizon-1]
            
            # Create features for this time window
            node_features = np.zeros((n_subfields, window_size * 3))
            
            # For each month in the input window, calculate features
            for j, month in enumerate(input_months):
                graph = self.monthly_graphs[month]
                
                # Features: degree, clustering coefficient, eigenvector centrality
                for k in range(n_subfields):
                    # Degree
                    node_features[k, j*3] = np.sum(graph[k])
                    
                    # Create NetworkX graph for other centrality measures
                    nx_graph = nx.from_numpy_array(graph)
                    
                    # Clustering coefficient
                    clustering = nx.clustering(nx_graph)
                    node_features[k, j*3+1] = clustering.get(k, 0)
                    
                    # Eigenvector centrality
                    try:
                        ev_centrality = nx.eigenvector_centrality(nx_graph, max_iter=1000)
                        node_features[k, j*3+2] = ev_centrality.get(k, 0)
                    except:
                        # If eigenvector centrality fails (e.g., for disconnected graphs)
                        node_features[k, j*3+2] = 0
            
            # Create edge index and labels
            edge_index = []
            edge_attr = []
            labels = []
            
            for src in range(n_subfields):
                for dst in range(src+1, n_subfields):
                    # Current collaboration strength across input window
                    collab_history = []
                    for month in input_months:
                        graph = self.monthly_graphs[month]
                        collab_history.append(graph[src, dst])
                    
                    # Target collaboration strength
                    target_graph = self.monthly_graphs[target_month]
                    target_value = target_graph[src, dst]
                    
                    # Only include edges that have some collaboration in the input window
                    if np.sum(collab_history) > 0:
                        edge_index.append([src, dst])
                        edge_index.append([dst, src])  # Undirected graph
                        
                        # Edge features: collaboration history
                        edge_attr.append(collab_history)
                        edge_attr.append(collab_history)  # Same for both directions
                        
                        # Labels: collaboration strength in target month
                        labels.append(target_value)
                        labels.append(target_value)  # Same for both directions
            
            if edge_index:  # Only add if there are edges
                # Convert to PyTorch tensors
                x = torch.FloatTensor(node_features)
                edge_index = torch.LongTensor(edge_index).t().contiguous()  # PyG expects shape [2, num_edges]
                edge_attr = torch.FloatTensor(edge_attr)
                y = torch.FloatTensor(labels)
                
                # Create PyG data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                
                self.training_data.append(data)
        
        print(f"Created {len(self.training_data)} temporal graph samples for training")
    
    def calculate_growth_metrics(self, window_size=6, prediction_horizon=6):
        """
        Calculate growth metrics for subfield collaborations.
        This analyzes historical data to show how collaborations have grown
        over time, which helps validate our model's predictions.
        
        Args:
            window_size (int): Number of months in the input window
            prediction_horizon (int): Number of months ahead to predict
        
        Returns:
            pandas.DataFrame: DataFrame with growth metrics
        """
        # Sort months chronologically
        sorted_months = sorted(self.monthly_graphs.keys())
        
        if len(sorted_months) < window_size + prediction_horizon:
            raise ValueError(f"Not enough data. Need at least {window_size + prediction_horizon} months.")
        
        growth_data = []
        
        # For each subfield pair
        n_subfields = len(self.subfield_mapping)
        for src in range(n_subfields):
            for dst in range(src+1, n_subfields):
                src_id = self.reverse_mapping[src]
                dst_id = self.reverse_mapping[dst]
                src_name = self.subfield_names[src_id]
                dst_name = self.subfield_names[dst_id]
                
                # For each valid time window
                for i in range(len(sorted_months) - window_size - prediction_horizon + 1):
                    input_months = sorted_months[i:i+window_size]
                    prediction_months = sorted_months[i+window_size:i+window_size+prediction_horizon]
                    
                    # Calculate input window metrics
                    input_collab_counts = []
                    for month in input_months:
                        graph = self.monthly_graphs[month]
                        input_collab_counts.append(graph[src, dst])
                    
                    # Calculate prediction window metrics
                    prediction_collab_counts = []
                    for month in prediction_months:
                        graph = self.monthly_graphs[month]
                        prediction_collab_counts.append(graph[src, dst])
                    
                    # Only include pairs with some collaboration
                    if np.sum(input_collab_counts) > 0 or np.sum(prediction_collab_counts) > 0:
                        input_count = np.sum(input_collab_counts)
                        prediction_count = np.sum(prediction_collab_counts)
                        
                        # Calculate growth metrics
                        absolute_growth = prediction_count - input_count
                        
                        # Handle division by zero
                        if input_count == 0:
                            percent_growth = 100 if prediction_count > 0 else 0
                        else:
                            percent_growth = ((prediction_count / input_count) - 1) * 100
                        
                        # Add to growth data
                        growth_data.append({
                            'source_id': src_id,
                            'source_name': src_name,
                            'target_id': dst_id,
                            'target_name': dst_name,
                            'input_window_start': input_months[0],
                            'input_window_end': input_months[-1],
                            'prediction_window_start': prediction_months[0],
                            'prediction_window_end': prediction_months[-1],
                            'input_count': input_count,
                            'prediction_count': prediction_count,
                            'absolute_growth': absolute_growth,
                            'percent_growth': percent_growth
                        })
        
        return pd.DataFrame(growth_data)


class TemporalGNN(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim=64, dropout_rate=0.2):
        """
        Temporal Graph Neural Network for predicting subfield collaborations.
        
        Args:
            input_node_dim (int): Dimension of node features
            input_edge_dim (int): Dimension of edge features
            hidden_dim (int): Dimension of hidden layers
            dropout_rate (float): Dropout rate for regularization
        """
        self.dropout_rate = dropout_rate
        super(TemporalGNN, self).__init__()
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph convolution layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        
        # Edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        
        # Predict edge weights (collaboration strength)
        edge_pred = []
        for i in range(0, edge_index.shape[1], 2):  # Process each undirected edge once
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_feat = torch.cat([x[src], x[dst]])
            pred = self.edge_predictor(edge_feat)
            edge_pred.append(pred)
            edge_pred.append(pred)  # Add same prediction for reverse edge
        
        # Stack predictions
        if edge_pred:
            edge_pred = torch.stack(edge_pred).squeeze()
        else:
            edge_pred = torch.tensor([])
        
        return edge_pred


def visualize_subfield_network(subfield_gnn, month):
    """
    Visualize the subfield collaboration network for a specific month.
    This helps understand how the collaboration network evolves over time.
    
    Args:
        subfield_gnn (SubfieldGNN): Data handler
        month (str): Month to visualize (format: 'YYYY-MM')
    """
    if month not in subfield_gnn.monthly_graphs:
        print(f"No data available for month {month}")
        return
    
    adj_matrix = subfield_gnn.monthly_graphs[month]
    G = nx.from_numpy_array(adj_matrix)
    
    # Map node indices to subfield names
    node_labels = {}
    for idx in range(len(subfield_gnn.reverse_mapping)):
        subfield_id = subfield_gnn.reverse_mapping[idx]
        node_labels[idx] = subfield_gnn.subfield_names[subfield_id]
    
    # Calculate node sizes based on total connections
    node_sizes = [np.sum(adj_matrix[i]) * 100 for i in range(adj_matrix.shape[0])]
    
    # Calculate edge weights
    edge_weights = [adj_matrix[u, v] for u, v in G.edges()]
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7)
    
    # Draw edges with varying thickness
    nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in edge_weights], alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title(f"Subfield Collaboration Network - {month}")
    plt.tight_layout()
    plt.axis('off')
    plt.show()