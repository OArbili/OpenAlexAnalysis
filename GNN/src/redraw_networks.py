#!/usr/bin/env python
# redraw_networks.py
# Script to redraw network visualizations from saved data

from save_load_network_data import load_and_draw_network
import argparse

def redraw_subfield_network(font_size=14, figsize=(20, 16), node_scale=1.2, edge_scale=1.5):
    """
    Redraw the subfield-level network with custom parameters
    
    Args:
        font_size (int): Font size for node labels
        figsize (tuple): Figure size as (width, height)
        node_scale (float): Scaling factor for node sizes
        edge_scale (float): Scaling factor for edge widths
    """
    print("Redrawing subfield-level network visualization...")
    
    load_and_draw_network(
        file_name='subfield_network_data.pkl',
        figsize=figsize,
        font_size=font_size,
        title='Interdisciplinary Collaboration Network (Top 50 Growing Collaborations)',
        output_file='redrawn_subfield_network.png',
        node_size_scale=node_scale,
        edge_width_scale=edge_scale
    )

def redraw_field_network(font_size=16, figsize=(18, 14), node_scale=1.2, edge_scale=2.0):
    """
    Redraw the field-level network with custom parameters
    
    Args:
        font_size (int): Font size for node labels
        figsize (tuple): Figure size as (width, height)
        node_scale (float): Scaling factor for node sizes
        edge_scale (float): Scaling factor for edge widths
    """
    print("Redrawing field-level network visualization...")
    
    load_and_draw_network(
        file_name='field_level_network_data.pkl',
        figsize=figsize,
        font_size=font_size,
        title='Field-Level Interdisciplinary Collaboration Network',
        output_file='redrawn_field_network.png',
        node_size_scale=node_scale,
        edge_width_scale=edge_scale
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Redraw network visualizations with custom parameters')
    parser.add_argument('--network', choices=['subfield', 'field', 'both'], default='both',
                        help='Which network to redraw (subfield, field, or both)')
    parser.add_argument('--font-size', type=int, default=14,
                        help='Font size for node labels')
    parser.add_argument('--width', type=int, default=20,
                        help='Figure width in inches')
    parser.add_argument('--height', type=int, default=16,
                        help='Figure height in inches')
    parser.add_argument('--node-scale', type=float, default=1.2,
                        help='Scaling factor for node sizes')
    parser.add_argument('--edge-scale', type=float, default=1.5,
                        help='Scaling factor for edge widths')
    
    args = parser.parse_args()
    figsize = (args.width, args.height)
    
    if args.network in ['subfield', 'both']:
        redraw_subfield_network(
            font_size=args.font_size,
            figsize=figsize,
            node_scale=args.node_scale,
            edge_scale=args.edge_scale
        )
    
    if args.network in ['field', 'both']:
        # For field network, use slightly larger font
        redraw_field_network(
            font_size=args.font_size + 2,
            figsize=figsize,
            node_scale=args.node_scale,
            edge_scale=args.edge_scale * 1.5  # Edges even thicker for field network
        )
    
    print("\nVisualization(s) redrawn successfully!")
    print("- Subfield network: redrawn_subfield_network.png")
    print("- Field network: redrawn_field_network.png")
    print("\nTip: To customize further, try different parameter combinations:")
    print("  python redraw_networks.py --network subfield --font-size 16 --width 24 --height 18 --node-scale 1.5 --edge-scale 2.0")