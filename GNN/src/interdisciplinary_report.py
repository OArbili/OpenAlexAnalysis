# interdisciplinary_report.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os
import torch 

def generate_interdisciplinary_report(model, subfield_gnn, top_k=30, debug=False):
    """
    Generate a comprehensive report on predicted interdisciplinary collaborations.
    Only includes relationships with current collaborations >= 1.
    
    Args:
        model: Trained GNN model
        subfield_gnn: SubfieldGNN data handler
        top_k: Number of top collaborations to include
        debug: Whether to run in debug mode (simplified visualizations)
    
    Returns:
        DataFrame with interdisciplinary growth predictions
    """
    print("Extracting interdisciplinary collaborations...")
    start_time = time.time()
    
    try:
        # Get interdisciplinary growth predictions with min_collaborations=3
        growth_df = get_interdisciplinary_growth_report(model, subfield_gnn, top_k=top_k, min_collaborations=3)
        
        if growth_df.empty:
            print("No interdisciplinary collaborations found.")
            return pd.DataFrame()
        
        print(f"Found {len(growth_df)} interdisciplinary collaborations")
        
        # In debug mode, create minimal visualizations
        if debug:
            print("DEBUG MODE: Creating simplified report")
            
            # Just save the data to CSV
            growth_df.to_csv("interdisciplinary_report_debug.csv", index=False)
            print("Debug report saved to 'interdisciplinary_report_debug.csv'")
            
            return growth_df
        
        # Create visualizations (only in full mode)
        print("Creating visualizations...")
        plt.figure(figsize=(12, 10))
        
        # 1. Bar chart of top interdisciplinary collaborations by absolute growth
        plt.subplot(2, 1, 1)
        top_10 = growth_df.head(10).copy()
        # Truncate long subfield names
        top_10['short_relationship'] = top_10['relationship'].apply(
            lambda x: x[:60] + '...' if len(x) > 60 else x
        )
        
        sns.barplot(x='absolute_growth', y='short_relationship', data=top_10)
        plt.title('Top 10 Interdisciplinary Collaborations by Absolute Growth')
        plt.xlabel('Predicted Absolute Growth')
        plt.ylabel('Subfield Pair')
        plt.tight_layout()
        
        # 2. Extract field pairs for heatmap
        plt.subplot(2, 1, 2)
        
        # Extract field pairs from the full dataframe
        field_pairs = []
        growth_values = []
        
        for _, row in growth_df.iterrows():
            relationship_parts = row['relationship'].split(' ↔ ')
            if len(relationship_parts) != 2:
                continue
                
            try:
                field1 = relationship_parts[0].split(' (')[-1][:-1]  # Extract first field
                field2 = relationship_parts[1].split(' (')[-1][:-1]  # Extract second field
                
                # Ensure consistent ordering of fields
                if field1 > field2:
                    field1, field2 = field2, field1
                    
                field_pairs.append((field1, field2))
                growth_values.append(row['absolute_growth'])
            except:
                continue
        
        # Create a DataFrame for the heatmap
        field_growth_df = pd.DataFrame({
            'field1': [fp[0] for fp in field_pairs],
            'field2': [fp[1] for fp in field_pairs],
            'growth': growth_values
        })
        
        if not field_growth_df.empty:
            # Aggregate by field pairs
            field_growth_pivot = field_growth_df.groupby(['field1', 'field2'])['growth'].sum().reset_index()
            
            if len(field_growth_pivot) > 1:  # Only create heatmap if we have multiple fields
                field_heatmap = field_growth_pivot.pivot_table(
                    index='field1', 
                    columns='field2', 
                    values='growth',
                    aggfunc='sum'
                ).fillna(0)
                
                # Plot heatmap
                sns.heatmap(field_heatmap, annot=True, cmap='viridis', fmt='.1f')
                plt.title('Interdisciplinary Collaboration Growth by Field Pairs')
        
        plt.tight_layout()
        plt.savefig('interdisciplinary_heatmap.png', dpi=300, bbox_inches='tight')
        
        # Generate tabular report
        report_text = "# Interdisciplinary Collaboration Growth Report\n\n"
        report_text += "## Top Growing Interdisciplinary Collaborations (Minimum 3 Current Collaborations)\n\n"
        
        # Format the table
        table_text = "| Rank | Collaboration | Current | Predicted | Absolute Growth | % Growth |\n"
        table_text += "|------|--------------|---------|-----------|----------------|----------|\n"
        
        for i, (_, row) in enumerate(growth_df.iterrows(), 1):
            # Parse the growth summary to extract current and predicted values
            try:
                growth_parts = row['growth_summary'].split('→')
                current = growth_parts[0].split(':')[1].strip()
                predicted_parts = growth_parts[1].split('(')
                predicted = predicted_parts[0].split(':')[1].strip()
                
                table_text += f"| {i} | {row['relationship']} | {current} | {predicted} | {row['absolute_growth']:.1f} | {row['percent_growth']:.1f}% |\n"
            except:
                # Fallback if parsing fails
                table_text += f"| {i} | {row['relationship']} | - | - | {row['absolute_growth']:.1f} | {row['percent_growth']:.1f}% |\n"
        
        report_text += table_text
        
        # Add interdisciplinary insights
        report_text += "\n## Key Insights\n\n"
        
        # Find fields with highest overall growth
        field_counts = {}
        for _, row in growth_df.iterrows():
            try:
                relationship_parts = row['relationship'].split(' ↔ ')
                field1 = relationship_parts[0].split(' (')[-1][:-1]
                field2 = relationship_parts[1].split(' (')[-1][:-1]
                
                field_counts[field1] = field_counts.get(field1, 0) + row['absolute_growth']
                field_counts[field2] = field_counts.get(field2, 0) + row['absolute_growth']
            except:
                continue
        
        # Sort fields by growth
        sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
        
        report_text += "### Most Collaborative Fields\n\n"
        report_text += "Fields showing highest interdisciplinary growth:\n\n"
        for field, growth in sorted_fields[:min(5, len(sorted_fields))]:
            report_text += f"- **{field}**: {growth:.1f} total predicted collaboration growth\n"
        
        # Add definitions to help understand the metrics
        report_text += "\n### Understanding Growth Metrics\n\n"
        report_text += "- **Current**: Number of collaboration instances observed in the most recent months\n"
        report_text += "- **Predicted**: Expected number of future collaboration instances\n"
        report_text += "- **Absolute Growth**: The numerical difference between current and predicted values\n"
        report_text += "- **% Growth**: Percentage increase in collaborations\n\n"
        
        # Add recommendations
        report_text += "\n### Recommendations\n\n"
        report_text += "Based on the predicted collaboration growth, consider:\n\n"
        if len(sorted_fields) >= 2:
            report_text += "1. Allocating more resources to interdisciplinary research between " + \
                          f"{sorted_fields[0][0]} and {sorted_fields[1][0]}\n"
        report_text += "2. Creating dedicated conferences or journals for emerging interdisciplinary areas\n"
        report_text += "3. Establishing joint funding initiatives across different fields\n"
        report_text += "4. Developing academic programs that bridge these growing interdisciplinary connections\n"
        
        # Save the report as a markdown file
        with open("interdisciplinary_report.md", "w") as f:
            f.write(report_text)
        
        # Also save as CSV for easy data access
        growth_df.to_csv("interdisciplinary_growth.csv", index=False)
        
        print("Interdisciplinary report generated successfully.")
        print("- Visualization saved to 'interdisciplinary_heatmap.png'")
        print("- Report saved to 'interdisciplinary_report.md'")
        print("- Data saved to 'interdisciplinary_growth.csv'")
        
        elapsed_time = time.time() - start_time
        print(f"Report generation completed in {elapsed_time:.2f} seconds")
        
        return growth_df
    
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

        
def visualize_top_interdisciplinary_collaborations(growth_df, subfield_gnn, debug=False):
    """
    Create a network visualization of the top interdisciplinary collaborations.
    
    Args:
        growth_df: DataFrame with interdisciplinary growth predictions
        subfield_gnn: SubfieldGNN data handler
        debug: Whether to run in debug mode (simplified visualization)
    """
    if growth_df.empty:
        print("No data to visualize.")
        return
    
    print("Creating network visualization...")
    start_time = time.time()
    
    try:
        import networkx as nx
        
        # In debug mode, create a very simplified network
        if debug:
            print("DEBUG MODE: Creating simplified network visualization")
            
            # Create a minimal graph
            G = nx.Graph()
            
            # Add just the top 5 connections
            for i, (_, row) in enumerate(growth_df.head(5).iterrows()):
                parts = row['relationship'].split(' ↔ ')
                if len(parts) == 2:
                    subfield1 = parts[0].split(' (')[0]
                    subfield2 = parts[1].split(' (')[0]
                    G.add_edge(subfield1, subfield2, weight=1)
            
            # Draw a simple graph
            plt.figure(figsize=(8, 6))
            nx.draw(G, with_labels=True, node_color='skyblue', font_size=10)
            plt.title('Simplified Interdisciplinary Network (Debug Mode)')
            plt.savefig('debug_network.png')
            
            print("Debug visualization saved to 'debug_network.png'")
            return
        
        # Full visualization mode
        # Create a graph
        G = nx.Graph()
        
        # Extract unique fields and subfields
        fields = set()
        subfields = {}
        
        # Use top 50 connections for a more comprehensive visualization
        visualize_df = growth_df.head(50)
        
        for _, row in visualize_df.iterrows():
            try:
                relationship_parts = row['relationship'].split(' ↔ ')
                if len(relationship_parts) != 2:
                    continue
                
                # Extract first subfield and field
                subfield1_parts = relationship_parts[0].split(' (')
                subfield1 = subfield1_parts[0]
                field1 = subfield1_parts[1][:-1]
                
                # Extract second subfield and field
                subfield2_parts = relationship_parts[1].split(' (')
                subfield2 = subfield2_parts[0]
                field2 = subfield2_parts[1][:-1]
                
                # Add to sets
                fields.add(field1)
                fields.add(field2)
                subfields[subfield1] = field1
                subfields[subfield2] = field2
                
                # Add nodes and edges
                if not G.has_node(subfield1):
                    G.add_node(subfield1, field=field1)
                if not G.has_node(subfield2):
                    G.add_node(subfield2, field=field2)
                    
                # Add edge with growth as weight
                G.add_edge(subfield1, subfield2, 
                           weight=row['absolute_growth'],
                           percent_growth=row['percent_growth'])
            except Exception as e:
                print(f"Error processing relationship: {e}")
                continue
        
        if len(G.nodes()) == 0:
            print("No valid nodes found for visualization.")
            return
        
        # Assign colors to fields
        field_colors = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
        
        for i, field in enumerate(fields):
            field_colors[field] = colors[i % len(colors)]
        
        # Assign node colors based on fields
        node_colors = []
        for node in G.nodes():
            field = G.nodes[node].get('field')
            color = field_colors.get(field, '#cccccc')  # Default gray if field not found
            node_colors.append(color)
        
        # Assign edge widths based on growth
        edge_weights = []
        for u, v in G.edges():
            weight = G[u][v].get('weight', 1)
            # Scale width logarithmically to prevent extremely thick edges
            edge_weights.append(max(0.5, min(3, 0.5 * np.log1p(weight))))
        
        # Create layout - try different algorithms for better visualization
        try:
            # Try spring layout with customized parameters to spread nodes
            pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
        except:
            try:
                # Try kamada_kawai layout which often gives good results for networks
                pos = nx.kamada_kawai_layout(G)
            except:
                # Fallback to simpler layout if others fail
                pos = nx.circular_layout(G)
        
        # Create a larger figure for better visibility with many nodes
        plt.figure(figsize=(18, 14))
        
        # Draw nodes with appropriate size
        node_sizes = [400 for _ in G.nodes()]  # Fixed size for better readability
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Draw edges with varying width
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')
        
        # Draw labels with even larger font to improve readability
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Create legend for fields
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=field, markerfacecolor=color, markersize=10)
                          for field, color in field_colors.items()]
        
        # Place legend outside the main plot with smaller font
        plt.legend(handles=legend_elements, title="Research Fields", 
                  loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=min(5, len(fields)),
                  fontsize=8)
        
        plt.title('Interdisciplinary Collaboration Network (Top 50 Growing Collaborations)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('interdisciplinary_network.png', dpi=300, bbox_inches='tight')
        
        # Save subfield-level network data for later redrawing
        from save_load_network_data import save_network_data
        
        save_network_data(
            graph=G,
            pos=pos,
            node_colors=node_colors,
            edge_weights=edge_weights,
            node_sizes=node_sizes,
            field_colors=field_colors,
            file_name='subfield_network_data.pkl'
        )
        
        # Also create a simplified version focusing on the field level
        print("Creating field-level network visualization...")
        
        # Create a field-level graph
        F = nx.Graph()
        
        # Add fields as nodes
        for field in fields:
            F.add_node(field)
        
        # Add edges between fields
        field_connections = {}
        for _, row in visualize_df.iterrows():
            try:
                relationship_parts = row['relationship'].split(' ↔ ')
                if len(relationship_parts) != 2:
                    continue
                
                field1 = relationship_parts[0].split(' (')[-1][:-1]
                field2 = relationship_parts[1].split(' (')[-1][:-1]
                
                # Ensure consistent ordering
                if field1 > field2:
                    field1, field2 = field2, field1
                
                key = f"{field1}--{field2}"
                if key not in field_connections:
                    field_connections[key] = 0
                
                field_connections[key] += row['absolute_growth']
            except:
                continue
        
        # Add edges to graph
        for key, weight in field_connections.items():
            field1, field2 = key.split('--')
            F.add_edge(field1, field2, weight=weight)
        
        # Create field-level visualization
        plt.figure(figsize=(14, 10))
        
        # Use a different layout for field graph
        field_pos = nx.spring_layout(F, k=0.5, seed=42)
        
        # Node colors
        field_node_colors = [field_colors.get(field, '#cccccc') for field in F.nodes()]
        
        # Node sizes based on total growth
        field_node_sizes = []
        for field in F.nodes():
            # Sum of all connections involving this field
            size = sum(weight for (f1, f2, weight) in F.edges(data='weight') if f1 == field or f2 == field)
            field_node_sizes.append(500 + size * 20)  # Base size + weight factor
        
        # Edge widths - make them wider for better visibility
        field_edge_widths = [F[u][v]['weight'] * 0.5 for u, v in F.edges()]
        
        # Draw the field graph
        nx.draw_networkx_nodes(F, field_pos, node_size=field_node_sizes, node_color=field_node_colors, alpha=0.8)
        nx.draw_networkx_edges(F, field_pos, width=field_edge_widths, alpha=0.7, edge_color='darkgray')
        nx.draw_networkx_labels(F, field_pos, font_size=14, font_weight='bold')
        
        # Add edge labels for connection strength
        edge_labels = {(u, v): f"{F[u][v]['weight']:.1f}" for u, v in F.edges()}
        nx.draw_networkx_edge_labels(F, field_pos, edge_labels=edge_labels, font_size=8)
        
        plt.title('Field-Level Interdisciplinary Collaboration Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('field_level_network.png', dpi=300, bbox_inches='tight')
        
        # Save field-level network data for later redrawing
        from save_load_network_data import save_network_data
        
        save_network_data(
            graph=F,
            pos=field_pos,
            node_colors=field_node_colors,
            edge_weights=field_edge_widths,
            node_sizes=field_node_sizes,
            field_colors={field: color for field, color in field_colors.items() if field in fields},
            file_name='field_level_network_data.pkl'
        )
        
        print("Interdisciplinary network visualizations saved to:")
        print("- 'interdisciplinary_network.png' (subfield level)")
        print("- 'field_level_network.png' (field level)")
        
        elapsed_time = time.time() - start_time
        print(f"Visualization completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        

# Modified version of get_interdisciplinary_growth_report function
def get_interdisciplinary_growth_report(model, subfield_gnn, top_k=40, min_collaborations=1):
    """
    Generate a report of the top growing collaborations between subfields
    that belong to different fields (truly interdisciplinary collaborations).
    Only includes relationships with current collaborations >= min_collaborations.
    
    Args:
        model: Trained GNN model
        subfield_gnn: Data handler
        top_k: Number of top predictions to return
        min_collaborations: Minimum number of current collaborations required (default: 1)
    
    Returns:
        pandas.DataFrame: DataFrame with top interdisciplinary growth predictions
    """
    print("Analyzing interdisciplinary collaborations...")
    
    try:
        # Check if model is a torch module
        is_torch_model = hasattr(model, 'eval') and callable(model.eval)
        
        if is_torch_model:
            model.eval()
            # Get device
            device = next(model.parameters()).device
        else:
            # For debug mode with dummy model
            device = torch.device('cpu')
        
        # Ensure we have training data
        if not hasattr(subfield_gnn, 'training_data') or len(subfield_gnn.training_data) == 0:
            print("No training data found.")
            return pd.DataFrame()
        
        # Get the latest data for prediction
        latest_data = subfield_gnn.training_data[-1].to(device)
        
        # Make predictions
        with torch.no_grad():
            if is_torch_model:
                predictions = model(latest_data).cpu().numpy()
            else:
                # For debug mode, just increase current values slightly
                predictions = latest_data.y.cpu().numpy() * 1.1
        
        # Create prediction results with field information
        results = []
        for i, edge_idx in enumerate(latest_data.edge_index.t().cpu().numpy()):
            if i % 2 == 0:  # Process each undirected edge once
                src, dst = edge_idx
                
                # Make sure indices are valid
                if src >= len(subfield_gnn.reverse_mapping) or dst >= len(subfield_gnn.reverse_mapping):
                    continue
                
                src_id = subfield_gnn.reverse_mapping[src]
                dst_id = subfield_gnn.reverse_mapping[dst]
                
                # Get subfield names
                src_name = subfield_gnn.subfield_names.get(src_id, f"Subfield {src}")
                dst_name = subfield_gnn.subfield_names.get(dst_id, f"Subfield {dst}")
                
                # Get field information
                src_field = None
                dst_field = None
                
                # Look through all articles to find field information
                for article in subfield_gnn.articles:
                    for topic in article.get('topics', []):
                        if 'subfield' in topic and topic['subfield']:
                            if topic['subfield']['id'] == src_id and 'field' in topic:
                                src_field = topic['field']['display_name']
                            if topic['subfield']['id'] == dst_id and 'field' in topic:
                                dst_field = topic['field']['display_name']
                
                # If we couldn't find fields, skip this pair
                if not src_field or not dst_field:
                    continue
                
                # Only include indices within range
                if i < len(predictions):
                    prediction = predictions[i]
                    
                    # Make sure we have a valid y value
                    if i < len(latest_data.y):
                        current = latest_data.y[i].item()
                    else:
                        current = 0
                    
                    # Only include if current collaborations >= min_collaborations and fields are different
                    if src_field != dst_field and current >= min_collaborations:
                        results.append({
                            'source_id': src_id,
                            'source_name': src_name,
                            'source_field': src_field,
                            'target_id': dst_id,
                            'target_name': dst_name,
                            'target_field': dst_field,
                            'predicted_collaborations': float(prediction),
                            'current_collaborations': float(current),
                            'absolute_growth': float(prediction) - float(current)
                        })
        
        # Convert to DataFrame and calculate percentage growth
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            # Calculate percentage growth, handling division by zero
            results_df['percent_growth'] = results_df.apply(
                lambda row: ((row['predicted_collaborations'] / row['current_collaborations']) - 1) * 100
                if row['current_collaborations'] > 0 else float('inf') if row['predicted_collaborations'] > 0 else 0,
                axis=1
            )
            
            # Sort by absolute growth
            results_df = results_df.sort_values('absolute_growth', ascending=False)
            
            # Format output for better readability
            results_df['relationship'] = results_df.apply(
                lambda row: f"{row['source_name']} ({row['source_field']}) ↔ "
                           f"{row['target_name']} ({row['target_field']})",
                axis=1
            )
            
            # Format growth numbers
            results_df['growth_summary'] = results_df.apply(
                lambda row: f"Current: {row['current_collaborations']:.1f} → "
                           f"Predicted: {row['predicted_collaborations']:.1f} "
                           f"(+{row['absolute_growth']:.1f}, {row['percent_growth']:.1f}%)",
                axis=1
            )
            
            # Select columns for the report
            report_df = results_df[['relationship', 'growth_summary', 'absolute_growth', 'percent_growth']]
            
            return report_df.head(top_k)
        else:
            print(f"No interdisciplinary collaborations with >= {min_collaborations} current collaborations found.")
            return pd.DataFrame(columns=['relationship', 'growth_summary', 'absolute_growth', 'percent_growth'])
    
    except Exception as e:
        print(f"Error in interdisciplinary report generation: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['relationship', 'growth_summary', 'absolute_growth', 'percent_growth'])