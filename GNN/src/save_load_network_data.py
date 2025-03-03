import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import json

def save_network_data(graph, pos, node_colors, edge_weights, 
                     node_sizes=None, field_colors=None, file_name='network_data.pkl'):
    """
    Save network graph data to a file for later redrawing
    
    Args:
        graph (networkx.Graph): The network graph object
        pos (dict): Node positions dictionary
        node_colors (list): List of node colors
        edge_weights (list): List of edge weights
        node_sizes (list, optional): List of node sizes
        field_colors (dict, optional): Mapping of fields to colors
        file_name (str): Name of the file to save data to
    """
    # Convert graph to dictionary representation
    graph_data = nx.readwrite.json_graph.node_link_data(graph)
    
    # Create a data dictionary with all visualization parameters
    data = {
        'graph_data': graph_data,
        'pos': {str(k): v for k, v in pos.items()},  # Convert node keys to strings for JSON compatibility
        'node_colors': node_colors,
        'edge_weights': edge_weights,
        'node_sizes': node_sizes,
        'field_colors': field_colors
    }
    
    # Save data to pickle file
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    
    print(f"Network data saved to {file_name}")
    
    # Also save a JSON version for interoperability
    json_file_name = file_name.replace('.pkl', '.json')
    
    # Convert numpy arrays and other non-serializable objects
    json_data = {
        'graph_data': graph_data,
        'pos': {str(k): [float(coord) for coord in v] for k, v in pos.items()},
        'node_colors': [str(c) for c in node_colors],
        'edge_weights': [float(w) for w in edge_weights],
        'node_sizes': [float(s) if s is not None else None for s in (node_sizes or [])],
        'field_colors': {k: str(v) for k, v in (field_colors or {}).items()}
    }
    
    with open(json_file_name, 'w') as file:
        json.dump(json_data, file, indent=2)
    
    print(f"Network data also saved as JSON to {json_file_name}")
    
    # Save a metadata CSV with node information for easy analysis
    nodes_df = pd.DataFrame([
        {
            'node_id': node,
            'node_name': node,
            'field': graph.nodes[node].get('field', 'Unknown'),
            'color': node_colors[i] if i < len(node_colors) else 'Unknown',
            'size': node_sizes[i] if node_sizes and i < len(node_sizes) else 'Unknown',
            'x_position': pos[node][0],
            'y_position': pos[node][1],
            'degree': graph.degree[node]
        }
        for i, node in enumerate(graph.nodes())
    ])
    
    csv_file_name = file_name.replace('.pkl', '_nodes.csv')
    nodes_df.to_csv(csv_file_name, index=False)
    print(f"Node metadata saved to {csv_file_name}")
    
    # Save edge information
    edges_df = pd.DataFrame([
        {
            'source': u,
            'target': v,
            'weight': graph[u][v].get('weight', 1.0),
            'percent_growth': graph[u][v].get('percent_growth', 0.0)
        }
        for u, v in graph.edges()
    ])
    
    edge_csv_file_name = file_name.replace('.pkl', '_edges.csv')
    edges_df.to_csv(edge_csv_file_name, index=False)
    print(f"Edge metadata saved to {edge_csv_file_name}")

def load_and_draw_network(file_name='network_data.pkl', 
                         figsize=(18, 14), 
                         font_size=12, 
                         title='Network Visualization',
                         output_file='redrawn_network.png',
                         node_size_scale=1.0,
                         edge_width_scale=1.0):
    """
    Load network data from file and redraw the visualization with customizable parameters
    
    Args:
        file_name (str): Name of the file containing saved network data
        figsize (tuple): Figure size as (width, height)
        font_size (int): Font size for node labels
        title (str): Title for the visualization
        output_file (str): Name of the output image file
        node_size_scale (float): Scaling factor for node sizes
        edge_width_scale (float): Scaling factor for edge widths
    """
    # Load data from file
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    
    # Extract components
    graph_data = data['graph_data']
    pos = {eval(k) if isinstance(k, str) else k: v for k, v in data['pos'].items()}
    node_colors = data['node_colors']
    edge_weights = data['edge_weights']
    node_sizes = data['node_sizes']
    field_colors = data['field_colors']
    
    # Rebuild graph
    G = nx.readwrite.json_graph.node_link_graph(graph_data)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes
    if node_sizes:
        scaled_node_sizes = [s * node_size_scale for s in node_sizes]
        nx.draw_networkx_nodes(G, pos, node_size=scaled_node_sizes, node_color=node_colors, alpha=0.8)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=500 * node_size_scale, node_color=node_colors, alpha=0.8)
    
    # Draw edges with scaled width
    scaled_edge_weights = [w * edge_width_scale for w in edge_weights]
    nx.draw_networkx_edges(G, pos, width=scaled_edge_weights, alpha=0.6, edge_color='gray')
    
    # Draw labels with customizable font size
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
    
    # Create legend if field colors are available
    if field_colors:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=field, markerfacecolor=color, markersize=10)
                          for field, color in field_colors.items()]
        
        plt.legend(handles=legend_elements, title="Research Fields", 
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=min(5, len(field_colors)),
                  fontsize=font_size)
    
    plt.title(title, fontsize=font_size+4)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"Visualization redrawn and saved to {output_file}")
    
    return G, pos, node_colors, edge_weights, node_sizes, field_colors