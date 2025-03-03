import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import seaborn as sns
from datetime import datetime
import numpy as np
import os
import pickle
import pandas as pd
import json

def analyze_academic_collaboration(file_path, output_dir=None, debug_mode=False, sample_size=100, 
                                 min_articles_per_author=1, min_citations=0):
    """
    Main function to run the complete collaboration analysis on OpenAlex data.
    
    Parameters:
    - file_path: Path to the academic article data file in OpenAlex JSON format
    - output_dir: Directory to save output files (optional)
    - debug_mode: If True, only process a small sample of articles
    - sample_size: Number of articles to process in debug mode
    - min_articles_per_author: Filter to include only authors with at least this many articles
    - min_citations: Filter to include only articles with at least this many citations
    
    Returns:
    - results: Dictionary with all analysis results
    """
    import time
    start_time = time.time()
    
    print("Starting academic collaboration analysis for OpenAlex data...")
    if debug_mode:
        print(f"DEBUG MODE: Processing only {sample_size} articles")
    if min_articles_per_author > 1:
        print(f"Filtering for authors with at least {min_articles_per_author} articles")
    if min_citations > 0:
        print(f"Filtering for articles with at least {min_citations} citations")
    
    # Prepare data
    df = prepare_data(file_path, debug_mode=debug_mode, sample_size=sample_size, 
                     min_articles_per_author=min_articles_per_author, min_citations=min_citations)
    
    data_time = time.time()
    print(f"Data preparation took {data_time - start_time:.2f} seconds")
    
    if len(df) == 0:
        print("No valid articles found after filtering. Try adjusting your filter criteria.")
        return None
        
    print(f"Analyzing {len(df)} articles spanning from {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")
    
    # Print sample of the prepared data for verification
    print("\nSample of processed data:")
    print(df[['title', 'field', 'main_writer']].head())
    print(f"\nNumber of unique authors: {len(set([author for authors in df['all_writers'] for author in authors]))}")
    print(f"Number of unique fields: {df['field'].nunique()}")
    
    # Build overall collaboration network
    overall_network = build_coauthorship_network(df)
    network_time = time.time()
    print(f"Network construction took {network_time - data_time:.2f} seconds")
    print(f"Built network with {len(overall_network.nodes())} authors and {len(overall_network.edges())} collaborations")
    
    # If dataset is very small, adjust the time interval
    if len(df) < 10:
        time_interval = '3M'  # 3-month intervals for small datasets
    else:
        time_interval = '6M'  # 6-month intervals for larger datasets
    
    # Temporal analysis
    temporal_stats = analyze_collaboration_over_time(df, interval=time_interval)
    temporal_time = time.time()
    print(f"Temporal analysis took {temporal_time - network_time:.2f} seconds")
    print(f"Completed temporal analysis across {len(temporal_stats)} time periods")
    
    # Field-based analysis
    field_stats = analyze_collaboration_by_field(df)
    field_time = time.time()
    print(f"Field analysis took {field_time - temporal_time:.2f} seconds")
    print(f"Completed field analysis across {len(field_stats)} fields")
    
    # Interdisciplinary analysis
    interdisciplinary_stats = analyze_interdisciplinary_collaboration(df)
    interdisciplinary_time = time.time()
    print(f"Interdisciplinary analysis took {interdisciplinary_time - field_time:.2f} seconds")
    print("Completed interdisciplinary analysis")
    
    # Key player analysis
    top_collaborators = identify_key_collaborators(df)
    key_player_time = time.time()
    print(f"Key player analysis took {key_player_time - interdisciplinary_time:.2f} seconds")
    print(f"Identified top {len(top_collaborators)} collaborators")
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Add debug or filter info to filenames if applicable
    file_prefix = ""
    if output_dir:
        if debug_mode:
            file_prefix = f"debug{sample_size}_"
        if min_articles_per_author > 1:
            file_prefix += f"min{min_articles_per_author}_"
        if min_citations > 0:
            file_prefix += f"cite{min_citations}_"

    # Initialize visualizations dictionary to store figures
    visualizations = {}
    
    # Make sure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Base path for data files (without extension)
    data_filepath_base = os.path.join(output_dir, f"{file_prefix}data") if output_dir else None
    
    # Limit network visualization to top authors if it's too large
    if len(overall_network.nodes()) > 100 and not debug_mode:
        print(f"Network is large ({len(overall_network.nodes())} nodes). Creating simplified visualization.")
        # Find top authors by degree centrality
        deg_centrality = nx.degree_centrality(overall_network)
        top_authors = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:50]
        top_author_names = [a[0] for a in top_authors]
        # Create subgraph with just the top authors
        viz_network = overall_network.subgraph(top_author_names)
        
        # Save network graph
        network_viz = plot_collaboration_network(
            viz_network, 
            title="Top Authors Co-authorship Network", 
            df=df,
            filename=os.path.join(output_dir, f"{file_prefix}collaboration_network.png") if output_dir else None,
            export_data=True,
            data_filename=os.path.join(output_dir, f"{file_prefix}top_authors") if output_dir else None
        )
    else:
        # Save full network graph
        network_viz = plot_collaboration_network(
            overall_network, 
            title="OpenAlex Co-authorship Network", 
            df=df,
            filename=os.path.join(output_dir, f"{file_prefix}collaboration_network.png") if output_dir else None,
            export_data=True,
            data_filename=os.path.join(output_dir, f"{file_prefix}full_network") if output_dir else None
        )
    
    visualizations['network'] = network_viz
    
    # Export temporal stats as raw data
    if output_dir:
        temporal_stats.to_csv(os.path.join(output_dir, f"{file_prefix}temporal_stats.csv"), index=False)
    
    # Only create trends visualization if we have enough time periods
    if len(temporal_stats) > 1:
        trends_viz = plot_collaboration_trends(
            temporal_stats,
            filename=os.path.join(output_dir, f"{file_prefix}collaboration_trends.png") if output_dir else None
        )
        visualizations['trends'] = trends_viz
    else:
        trends_viz = None
        visualizations['trends'] = None
        print("Not enough time periods for meaningful trend analysis")
    
    # Export field stats as raw data
    if output_dir:
        field_stats.to_csv(os.path.join(output_dir, f"{file_prefix}field_stats.csv"), index=False)
    
    # Only create field comparison if we have multiple fields
    if len(field_stats) > 1:
        field_viz = plot_field_comparison(
            field_stats,
            filename=os.path.join(output_dir, f"{file_prefix}field_comparison.png") if output_dir else None
        )
        visualizations['fields'] = field_viz
    else:
        field_viz = None
        visualizations['fields'] = None
        print("Not enough fields for meaningful field comparison")
    
    viz_time = time.time()
    print(f"Visualization generation took {viz_time - key_player_time:.2f} seconds")
    
    # Compile results before trying to save them
    results = {
        'overall_network': overall_network,
        'original_data': df,  # Store original data for visualization regeneration
        'temporal_stats': temporal_stats,
        'field_stats': field_stats,
        'interdisciplinary_stats': interdisciplinary_stats,
        'top_collaborators': top_collaborators,
        'analysis_params': {
            'file_path': file_path,
            'debug_mode': debug_mode,
            'sample_size': sample_size,
            'min_articles_per_author': min_articles_per_author,
            'min_citations': min_citations
        },
        'visualizations': visualizations
    }
    
    # Save results if output directory is provided
    if output_dir:
        # Save top collaborators data
        top_collaborators.to_csv(os.path.join(output_dir, f"{file_prefix}top_collaborators.csv"), index=False)
        
        # Save additional network data
        with open(os.path.join(output_dir, f"{file_prefix}collaboration_network.pkl"), 'wb') as f:
            pickle.dump(overall_network, f)
            
        # Save interdisciplinary stats
        with open(os.path.join(output_dir, f"{file_prefix}interdisciplinary_stats.json"), 'w') as f:
            json.dump(interdisciplinary_stats, f, indent=2)
            
        # Save analysis parameters
        with open(os.path.join(output_dir, f"{file_prefix}analysis_params.txt"), 'w') as f:
            f.write(f"Analysis parameters:\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Debug mode: {'Yes' if debug_mode else 'No'}\n")
            if debug_mode:
                f.write(f"Sample size: {sample_size}\n")
            f.write(f"Min articles per author: {min_articles_per_author}\n")
            f.write(f"Min citations per article: {min_citations}\n")
            f.write(f"Total articles analyzed: {len(df)}\n")
            f.write(f"Total authors: {len(overall_network.nodes())}\n")
            f.write(f"Total collaborations: {len(overall_network.edges())}\n")
            
        # Save results object for future visualization regeneration
        with open(os.path.join(output_dir, f"{file_prefix}analysis_results.pkl"), 'wb') as f:
            # Create a copy without matplotlib figures (which can't be pickled easily)
            results_to_save = results.copy()
            results_to_save['visualizations'] = None
            pickle.dump(results_to_save, f)
        
        # Also save a sample of the original DataFrame for reference
        try:
            sample_df = df.sample(min(1000, len(df)))
            sample_df.to_csv(os.path.join(output_dir, f"{file_prefix}data_sample.csv"), index=False)
        except Exception as e:
            print(f"Warning: Could not save data sample: {e}")
        
        print(f"Results saved to {output_dir}")
    
    # Add OpenAlex specific insights
    if 'author_ids' in df.columns:
        # Get frequency of collaborations by author pair
        author_pair_counts = {}
        for _, row in df.iterrows():
            authors = row['all_writers']
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    pair = tuple(sorted([authors[i], authors[j]]))
                    author_pair_counts[pair] = author_pair_counts.get(pair, 0) + 1
        
        # Sort by frequency
        frequent_collaborators = sorted(author_pair_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add to results
        results['frequent_collaborator_pairs'] = frequent_collaborators[:10]  # Top 10 most frequent pairs
        
        # Export top collaborative pairs
        if output_dir:
            with open(os.path.join(output_dir, f"{file_prefix}top_collaborator_pairs.csv"), 'w') as f:
                f.write("author1,author2,collaborations\n")
                for (a1, a2), count in frequent_collaborators[:50]:
                    f.write(f'"{a1}","{a2}",{count}\n')
        
        print("\nTop collaborative pairs:")
        for pair, count in frequent_collaborators[:5]:
            print(f"  {pair[0]} and {pair[1]}: {count} collaborations")
    
    end_time = time.time()
    print(f"\nTotal analysis time: {end_time - start_time:.2f} seconds")
    print("\nAnalysis complete!")
    return results

def regenerate_visualizations(results, output_dir=None, file_prefix=""):
    """
    Regenerate visualizations from saved analysis results without re-running the full analysis.
    
    Parameters:
    - results: Dictionary with analysis results
    - output_dir: Directory to save output files (optional)
    - file_prefix: Prefix for saved files
    
    Returns:
    - visualizations: Dictionary with regenerated visualization figures
    """
    print("Regenerating visualizations from analysis results...")
    
    # Extract data from results
    overall_network = results['overall_network']
    df = results.get('original_data')
    temporal_stats = results['temporal_stats']
    field_stats = results['field_stats']
    
    # Initialize visualizations dictionary
    visualizations = {}
    
    # Regenerate network visualization
    if len(overall_network.nodes()) > 100:
        print(f"Network is large ({len(overall_network.nodes())} nodes). Creating simplified visualization.")
        # Find top authors by degree centrality
        deg_centrality = nx.degree_centrality(overall_network)
        top_authors = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:50]
        top_author_names = [a[0] for a in top_authors]
        # Create subgraph with just the top authors
        viz_network = overall_network.subgraph(top_author_names)
        network_viz = plot_collaboration_network(
            viz_network, 
            title="Top Authors Co-authorship Network", 
            df=df,
            filename=os.path.join(output_dir, f"{file_prefix}collaboration_network.png") if output_dir else None
        )
    else:
        network_viz = plot_collaboration_network(
            overall_network, 
            title="Co-authorship Network", 
            df=df,
            filename=os.path.join(output_dir, f"{file_prefix}collaboration_network.png") if output_dir else None
        )
    
    visualizations['network'] = network_viz
    
    # Regenerate trends visualization if we have enough time periods
    if len(temporal_stats) > 1:
        trends_viz = plot_collaboration_trends(
            temporal_stats, 
            filename=os.path.join(output_dir, f"{file_prefix}collaboration_trends.png") if output_dir else None
        )
        visualizations['trends'] = trends_viz
    else:
        visualizations['trends'] = None
        print("Not enough time periods for meaningful trend analysis")
    
    # Regenerate field comparison if we have multiple fields
    if len(field_stats) > 1:
        field_viz = plot_field_comparison(
            field_stats, 
            filename=os.path.join(output_dir, f"{file_prefix}field_comparison.png") if output_dir else None
        )
        visualizations['fields'] = field_viz
    else:
        visualizations['fields'] = None
        print("Not enough fields for meaningful field comparison")
    
    print("Visualization regeneration complete!")
    return visualizations


# 1. DATA PREPARATION
def prepare_data(file_path, debug_mode=False, sample_size=100, min_articles_per_author=1, min_citations=0):
    """
    Load and prepare academic article data from OpenAlex JSON format.
    
    Parameters:
    - file_path: Path to the OpenAlex JSON file
    - debug_mode: If True, only process a small sample of articles
    - sample_size: Number of articles to process in debug mode
    - min_articles_per_author: Filter to include only authors with at least this many articles
    - min_citations: Filter to include only articles with at least this many citations
    
    Returns:
    - DataFrame with processed article data
    """
    # Load JSON data
    import json
    
    articles = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if debug_mode and i >= sample_size:
                break
                
            try:
                articles.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error parsing JSON line: {line[:50]}...")
    
    print(f"Loaded {len(articles)} articles from JSON file")
    
    # Filter articles by citation count if needed
    if min_citations > 0:
        filtered_articles = []
        for article in articles:
            citation_count = article.get('cited_by_count', 0)
            if citation_count >= min_citations:
                filtered_articles.append(article)
        
        print(f"Filtered from {len(articles)} to {len(filtered_articles)} articles with {min_citations}+ citations")
        articles = filtered_articles
    
    # Create a DataFrame with the necessary information
    processed_articles = []
    
    # First pass: collect all authors and their article counts
    author_article_counts = {}
    
    if min_articles_per_author > 1:
        print("First pass: counting articles per author...")
        for article in articles:
            if 'authorships' in article and article['authorships']:
                for authorship in article['authorships']:
                    if 'author' in authorship and 'display_name' in authorship['author']:
                        author_name = authorship['author']['display_name']
                        author_article_counts[author_name] = author_article_counts.get(author_name, 0) + 1
        
        print(f"Found {len(author_article_counts)} unique authors")
        print(f"Authors with {min_articles_per_author}+ articles: {sum(1 for count in author_article_counts.values() if count >= min_articles_per_author)}")
    
    # Second pass: process articles
    for article in articles:
        # Extract basic article info
        article_data = {
            'title': article.get('display_name', 'Unknown Title'),
            'publication_date': article.get('publication_date', None),
            'citations': article.get('cited_by_count', 0),
        }
        
        # Convert publication date to datetime
        if article_data['publication_date']:
            article_data['date'] = pd.to_datetime(article_data['publication_date'])
            article_data['year'] = article_data['date'].year
            article_data['month'] = article_data['date'].month
        else:
            continue  # Skip articles without publication date
            
        # Extract field and subfield from the first topic
        if 'topics' in article and article['topics']:
            main_topic = article['topics'][0]
            if 'field' in main_topic and main_topic['field']:
                article_data['field'] = main_topic['field'].get('display_name', 'Unknown Field')
            else:
                article_data['field'] = 'Unknown Field'
                
            if 'subfield' in main_topic and main_topic['subfield']:
                article_data['subfield'] = main_topic['subfield'].get('display_name', 'Unknown Subfield')
            else:
                article_data['subfield'] = 'Unknown Subfield'
                
            # Extract all topics as a list
            article_data['topics'] = [topic.get('display_name', 'Unknown Topic') for topic in article['topics']]
        else:
            article_data['field'] = 'Unknown Field'
            article_data['subfield'] = 'Unknown Subfield'
            article_data['topics'] = []
            
        # Extract author information
        if 'authorships' in article and article['authorships']:
            # Filter authors if min_articles_per_author > 1
            qualified_authors = []
            if min_articles_per_author > 1:
                for authorship in article['authorships']:
                    if 'author' in authorship and 'display_name' in authorship['author']:
                        author_name = authorship['author']['display_name']
                        if author_article_counts.get(author_name, 0) >= min_articles_per_author:
                            qualified_authors.append(authorship)
                
                # Skip this article if it has no qualified authors
                if not qualified_authors:
                    continue
            else:
                qualified_authors = article['authorships']
            
            # Find the main author (first or corresponding)
            main_author = None
            for authorship in qualified_authors:
                if authorship.get('author_position') == 'first' or authorship.get('is_corresponding', False):
                    if 'author' in authorship and 'display_name' in authorship['author']:
                        main_author = authorship['author']['display_name']
                        break
            
            if not main_author and qualified_authors:
                # If no first/corresponding author found, use the first author in the list
                if 'author' in qualified_authors[0] and 'display_name' in qualified_authors[0]['author']:
                    main_author = qualified_authors[0]['author']['display_name']
            
            article_data['main_writer'] = main_author or 'Unknown Author'
            
            # Extract all qualified authors as a list
            all_authors = []
            for authorship in qualified_authors:
                if 'author' in authorship and 'display_name' in authorship['author']:
                    all_authors.append(authorship['author']['display_name'])
            
            article_data['all_writers'] = all_authors
            
            # Skip articles with fewer than 2 authors
            if len(all_authors) < 2:
                continue
            
            # Extract author IDs for additional analysis
            author_ids = []
            for authorship in qualified_authors:
                if 'author' in authorship and 'id' in authorship['author']:
                    author_ids.append(authorship['author']['id'])
            article_data['author_ids'] = author_ids
            
        else:
            # Skip articles without authors
            continue
        
        processed_articles.append(article_data)
    
    # Create DataFrame
    df = pd.DataFrame(processed_articles)
    
    # Sort by date
    if 'date' in df.columns and not df.empty:
        df = df.sort_values('date')
    
    print(f"Processed {len(df)} articles after filtering")
    return df


# 2. NETWORK ANALYSIS
def build_coauthorship_network(df, time_window=None):
    """
    Build a collaboration network where nodes are authors and edges represent co-authorship.
    
    Parameters:
    - df: DataFrame with article data
    - time_window: Optional tuple (start_date, end_date) to filter data by time period
    
    Returns:
    - G: NetworkX graph of collaboration network
    """
    # Filter by time window if specified
    if time_window:
        start_date, end_date = time_window
        df_window = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        df_window = df
    
    # Create empty graph
    G = nx.Graph()
    
    # Process each paper
    for idx, row in df_window.iterrows():
        if 'all_writers' in df.columns:
            authors = row['all_writers']
            
            # Add all pairs of co-authors
            if isinstance(authors, list) and len(authors) > 1:
                for i in range(len(authors)):
                    for j in range(i+1, len(authors)):
                        author1, author2 = authors[i], authors[j]
                        
                        # Add nodes if they don't exist
                        if author1 not in G:
                            G.add_node(author1)
                        if author2 not in G:
                            G.add_node(author2)
                        
                        # Add or update edge
                        if G.has_edge(author1, author2):
                            G[author1][author2]['weight'] += 1
                        else:
                            G.add_edge(author1, author2, weight=1)
        else:
            # If only main_writer is available, we can't build a full co-authorship network
            G.add_node(row['main_writer'])
    
    return G

# 3. TEMPORAL ANALYSIS
def analyze_collaboration_over_time(df, interval='6M'):
    """
    Analyze how collaboration patterns change over time.
    
    Parameters:
    - df: DataFrame with article data
    - interval: Time interval for grouping (e.g., '6M' for 6 months, '1Y' for 1 year)
    
    Returns:
    - temporal_stats: DataFrame with temporal collaboration statistics
    """
    # Create time bins
    time_bins = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq=interval)
    
    # Initialize results storage
    temporal_stats = []
    
    # Loop through time periods
    for i in range(len(time_bins)-1):
        start_date = time_bins[i]
        end_date = time_bins[i+1]
        
        # Filter data for this time period
        period_data = df[(df['date'] >= start_date) & (df['date'] < end_date)]
        
        # Skip empty periods
        if len(period_data) == 0:
            continue
        
        # Build network for this period
        G = build_coauthorship_network(period_data)
        
        # Extract network metrics
        if len(G.nodes()) > 0:
            avg_clustering = nx.average_clustering(G)
            if len(G.nodes()) > 1:  # At least 2 nodes needed for these metrics
                density = nx.density(G)
                
                # Count connected components
                connected_components = list(nx.connected_components(G))
                largest_component_size = len(max(connected_components, key=len)) if connected_components else 0
                
                # Calculate average degree
                degrees = dict(G.degree())
                avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
            else:
                density = 0
                largest_component_size = 1 if len(G.nodes()) == 1 else 0
                avg_degree = 0
        else:
            avg_clustering = 0
            density = 0
            largest_component_size = 0
            avg_degree = 0
        
        # Store results
        temporal_stats.append({
            'period_start': start_date,
            'period_end': end_date,
            'num_papers': len(period_data),
            'num_authors': len(G.nodes()),
            'num_collaborations': len(G.edges()),
            'network_density': density,
            'avg_clustering': avg_clustering,
            'largest_component_size': largest_component_size,
            'avg_degree': avg_degree
        })
    
    return pd.DataFrame(temporal_stats)

# 4. FIELD-BASED ANALYSIS
def analyze_collaboration_by_field(df):
    """
    Analyze collaboration patterns by academic field and subfield.
    
    Parameters:
    - df: DataFrame with article data
    
    Returns:
    - field_stats: DataFrame with field-based collaboration statistics
    """
    # Initialize results storage
    field_stats = []
    
    # Analyze by field
    for field in df['field'].unique():
        field_data = df[df['field'] == field]
        
        # Create author collaboration network for this field
        G = build_coauthorship_network(field_data)
        
        # Get basic stats
        num_papers = len(field_data)
        num_authors = len(G.nodes())
        num_collaborations = len(G.edges())
        
        # Calculate network metrics if we have authors
        if num_authors > 0:
            avg_clustering = nx.average_clustering(G)
            if num_authors > 1:  # Need at least 2 authors for these metrics
                density = nx.density(G)
                
                # Find largest connected component
                connected_components = list(nx.connected_components(G))
                largest_component_size = len(max(connected_components, key=len)) if connected_components else 0
                
                # Calculate community modularity using Louvain method
                if len(G.edges()) > 0:
                    communities = nx.community.louvain_communities(G)
                    num_communities = len(communities)
                else:
                    num_communities = num_authors  # Each author is its own community
            else:
                density = 0
                largest_component_size = 1 if num_authors == 1 else 0
                num_communities = 1
        else:
            avg_clustering = 0
            density = 0
            largest_component_size = 0
            num_communities = 0
        
        # Store results
        field_stats.append({
            'field': field,
            'num_papers': num_papers,
            'num_authors': num_authors,
            'num_collaborations': num_collaborations,
            'network_density': density if num_authors > 1 else 0,
            'avg_clustering': avg_clustering,
            'largest_component_size': largest_component_size,
            'num_communities': num_communities
        })
    
    return pd.DataFrame(field_stats)

# 5. INTERDISCIPLINARY ANALYSIS
def analyze_interdisciplinary_collaboration(df):
    """
    Analyze interdisciplinary collaboration patterns.
    
    Parameters:
    - df: DataFrame with article data
    
    Returns:
    - interdisciplinary_stats: DataFrame with interdisciplinary metrics
    """
    # Map authors to their fields
    author_fields = {}
    
    # Process all papers
    for idx, row in df.iterrows():
        field = row['field']
        
        # Get all authors of the paper
        if 'all_writers' in df.columns and isinstance(row['all_writers'], list):
            authors = row['all_writers']
        else:
            authors = [row['main_writer']]
        
        # Update author-field mapping
        for author in authors:
            if author not in author_fields:
                author_fields[author] = set()
            author_fields[author].add(field)
    
    # Build overall collaboration network
    G = build_coauthorship_network(df)
    
    # Analyze interdisciplinary edges
    interdisciplinary_edges = 0
    total_edges = len(G.edges())
    
    for u, v in G.edges():
        u_fields = author_fields.get(u, set())
        v_fields = author_fields.get(v, set())
        
        # If authors have different fields, it's an interdisciplinary collaboration
        if len(u_fields.intersection(v_fields)) < len(u_fields.union(v_fields)):
            interdisciplinary_edges += 1
    
    # Calculate interdisciplinary ratio
    interdisciplinary_ratio = interdisciplinary_edges / total_edges if total_edges > 0 else 0
    
    # Count multi-field authors
    multi_field_authors = sum(1 for fields in author_fields.values() if len(fields) > 1)
    
    return {
        'total_authors': len(author_fields),
        'multi_field_authors': multi_field_authors,
        'multi_field_ratio': multi_field_authors / len(author_fields) if author_fields else 0,
        'interdisciplinary_collaborations': interdisciplinary_edges,
        'total_collaborations': total_edges,
        'interdisciplinary_ratio': interdisciplinary_ratio
    }

# 6. KEY PLAYER ANALYSIS
def identify_key_collaborators(df):
    """
    Identify key collaborators in the network using centrality measures and enhanced practical metrics.
    
    Parameters:
    - df: DataFrame with article data
    
    Returns:
    - top_authors: DataFrame with top authors by different centrality metrics and practical collaboration measures
    """
    # Build network
    G = build_coauthorship_network(df)
    
    # Skip if network is empty
    if len(G.nodes()) == 0:
        return pd.DataFrame()
    
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Calculate additional practical metrics
    
    # 1. Collaboration span (number of unique co-authors)
    collab_span = {}
    for author in G.nodes():
        collab_span[author] = len(list(G.neighbors(author)))
    
    # 2. Cross-field collaboration count
    cross_field_collabs = {}
    author_fields = {}
    
    # Map authors to their fields
    for _, row in df.iterrows():
        field = row['field']
        authors = row['all_writers']
        for author in authors:
            if author not in author_fields:
                author_fields[author] = set()
            author_fields[author].add(field)
    
    # Calculate cross-field collaboration for each author
    for author in G.nodes():
        author_field_set = author_fields.get(author, set())
        cross_field_count = 0
        
        # Count collaborations with authors from different fields
        for neighbor in G.neighbors(author):
            neighbor_field_set = author_fields.get(neighbor, set())
            if author_field_set != neighbor_field_set and len(author_field_set.intersection(neighbor_field_set)) < len(author_field_set.union(neighbor_field_set)):
                cross_field_count += 1
        
        cross_field_collabs[author] = cross_field_count
    
    # 3. Weighted degree (accounting for collaboration strength)
    weighted_degree = {}
    for author in G.nodes():
        weighted_sum = sum(G[author][neighbor].get('weight', 1) for neighbor in G.neighbors(author))
        weighted_degree[author] = weighted_sum
    
    # 4. Publication count per author
    publication_counts = {}
    for _, row in df.iterrows():
        for author in row['all_writers']:
            publication_counts[author] = publication_counts.get(author, 0) + 1
    
    # 5. Citation impact (total citations associated with each author)
    citation_impact = {}
    for _, row in df.iterrows():
        citations = row['citations']
        for author in row['all_writers']:
            citation_impact[author] = citation_impact.get(author, 0) + citations
    
    # Combine results
    author_data = []
    for author in G.nodes():
        author_data.append({
            'author': author,
            'degree_centrality': degree_centrality.get(author, 0),
            'betweenness_centrality': betweenness_centrality.get(author, 0),
            'eigenvector_centrality': eigenvector_centrality.get(author, 0),
            'collaboration_span': collab_span.get(author, 0),
            'cross_field_collaborations': cross_field_collabs.get(author, 0),
            'weighted_collaboration_strength': weighted_degree.get(author, 0),
            'publication_count': publication_counts.get(author, 0),
            'citation_impact': citation_impact.get(author, 0)
        })
    
    centrality_df = pd.DataFrame(author_data)
    
    # Calculate overall influence score (weighted average of metrics)
    centrality_df['influence_score'] = (
        0.20 * centrality_df['degree_centrality'] + 
        0.20 * centrality_df['betweenness_centrality'] + 
        0.15 * centrality_df['eigenvector_centrality'] +
        0.15 * (centrality_df['cross_field_collaborations'] / centrality_df['cross_field_collaborations'].max() if centrality_df['cross_field_collaborations'].max() > 0 else 0) +
        0.15 * (centrality_df['weighted_collaboration_strength'] / centrality_df['weighted_collaboration_strength'].max() if centrality_df['weighted_collaboration_strength'].max() > 0 else 0) +
        0.15 * (centrality_df['citation_impact'] / centrality_df['citation_impact'].max() if centrality_df['citation_impact'].max() > 0 else 0)
    )
    
    # Sort by influence score
    centrality_df = centrality_df.sort_values('influence_score', ascending=False)
    
    # Add collaboration diversity metric (ratio of cross-field to total collaborations)
    centrality_df['collaboration_diversity'] = centrality_df.apply(
        lambda row: row['cross_field_collaborations'] / row['collaboration_span'] if row['collaboration_span'] > 0 else 0, 
        axis=1
    )
    
    # Format centrality values to be more human-readable
    for col in ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'influence_score', 'collaboration_diversity']:
        centrality_df[col] = centrality_df[col].round(3)
    
    return centrality_df.head(20)  # Return top 20 influencers

# 7. VISUALIZATIONS
def plot_collaboration_network(G, title="Co-authorship Network", layout=nx.spring_layout, min_edge_weight=1, df=None, figsize=(10, 8), node_size_factor=60, filename=None, export_data=True, data_filename=None):
    """
    Visualize the collaboration network with improved readability and field-based coloring.
    
    Parameters:
    - G: NetworkX graph of the collaboration network
    - title: Plot title
    - layout: Network layout algorithm
    - min_edge_weight: Minimum edge weight to display
    - df: Original DataFrame with author field information
    - figsize: Figure size as tuple (width, height)
    - node_size_factor: Factor to scale node sizes
    - filename: If provided, save the figure to this file
    - export_data: If True, export network data for later visualization
    - data_filename: Filename for exported network data (without extension)
    
    Returns:
    - fig: The matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Filter edges by weight if needed
    if min_edge_weight > 1:
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1) >= min_edge_weight]
        G_filtered = G.edge_subgraph(edges_to_draw)
    else:
        G_filtered = G
    
    # Calculate node positions with more spacing
    pos = layout(G_filtered, k=1.5)  # k parameter increases spacing between nodes
    
    # Get degree for node size (scaled down)
    node_size = [v * node_size_factor for v in dict(G_filtered.degree()).values()]
    
    # Set edge width MUCH thinner
    edge_width = [0.1 + (d.get('weight', 1) * 0.05) for u, v, d in G_filtered.edges(data=True)]
    
    # Create a black-to-gray color range for edges based on weight
    max_weight = max([d.get('weight', 1) for u, v, d in G_filtered.edges(data=True)]) if G_filtered.edges() else 1
    edge_color = [(0.3 + (0.7 * d.get('weight', 1) / max_weight), 
                   0.3 + (0.7 * d.get('weight', 1) / max_weight), 
                   0.3 + (0.7 * d.get('weight', 1) / max_weight)) 
                 for u, v, d in G_filtered.edges(data=True)]
    
    # Determine node colors based on field if DataFrame is provided
    if df is not None:
        # Create a mapping of authors to their primary field
        author_fields = {}
        field_counts = {}
        
        # Count field occurrences for each author
        for _, row in df.iterrows():
            field = row['field']
            for author in row['all_writers']:
                if author not in author_fields:
                    author_fields[author] = {}
                author_fields[author][field] = author_fields[author].get(field, 0) + 1
        
        # Assign the most common field to each author
        author_primary_field = {}
        for author, fields in author_fields.items():
            primary_field = max(fields.items(), key=lambda x: x[1])[0]
            author_primary_field[author] = primary_field
            
            # Count unique fields for color mapping
            field_counts[primary_field] = field_counts.get(primary_field, 0) + 1
        
        # Create color map for fields (use a colorful palette)
        unique_fields = list(field_counts.keys())
        color_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_fields)))
        field_colors = {field: color_palette[i] for i, field in enumerate(unique_fields)}
        
        # Map each node to its color
        node_colors = [field_colors.get(author_primary_field.get(node, 'Unknown'), 'lightgray') for node in G_filtered.nodes()]
        
        # Create a legend for field colors
        field_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                  markersize=8, label=field) 
                      for field, color in field_colors.items()]
    else:
        # Default node colors if no field information
        node_colors = ['skyblue' for _ in G_filtered.nodes()]
        field_patches = []
    
    # Draw the network
    nx.draw_networkx_nodes(G_filtered, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G_filtered, pos, width=edge_width, alpha=0.6, edge_color=edge_color)
    
    # Use abbreviated labels if there are many nodes
    if len(G_filtered.nodes()) > 20:
        # Create abbreviated labels (first initial + last name)
        labels = {}
        for node in G_filtered.nodes():
            parts = node.split()
            if len(parts) > 1:
                labels[node] = f"{parts[0][0]}. {parts[-1]}"
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(G_filtered, pos, labels=labels, font_size=6, font_family='sans-serif')
    else:
        nx.draw_networkx_labels(G_filtered, pos, font_size=7, font_family='sans-serif')
    
    plt.title(title, fontsize=12)
    plt.axis('off')
    
    # Add field legend if available
    if field_patches:
        plt.legend(handles=field_patches, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Export network data for later visualization if requested
    if export_data and data_filename:
        network_data = {
            'nodes': [],
            'edges': [],
            'title': title,
            'author_fields': {}
        }
        
        # Save node data
        for node in G_filtered.nodes():
            degree = G_filtered.degree(node)
            field = author_primary_field.get(node, 'Unknown') if df is not None else 'Unknown'
            
            network_data['nodes'].append({
                'id': node,
                'label': labels.get(node, node) if len(G_filtered.nodes()) > 20 else node,
                'degree': degree,
                'field': field
            })
            
            # Save author-field mapping
            if df is not None and node in author_primary_field:
                network_data['author_fields'][node] = author_primary_field[node]
        
        # Save edge data
        for u, v, data in G_filtered.edges(data=True):
            weight = data.get('weight', 1)
            network_data['edges'].append({
                'source': u,
                'target': v,
                'weight': weight
            })
        
        # Save field color mapping
        if df is not None:
            network_data['field_colors'] = {
                field: list(color_rgb) for field, color_rgb in field_colors.items()
            }
        
        # Save node positions
        network_data['positions'] = {
            node: [float(pos[node][0]), float(pos[node][1])] for node in G_filtered.nodes()
        }
        
        # Save the data
        import json
        data_file = f"{data_filename}_network_data.json"
        with open(data_file, 'w') as f:
            json.dump(network_data, f, indent=2)
    
    return fig

def plot_collaboration_trends(temporal_stats, figsize=(12, 8), filename=None):
    """
    Plot trends of collaboration metrics over time.
    
    Parameters:
    - temporal_stats: DataFrame with temporal collaboration statistics
    - figsize: Figure size as tuple (width, height)
    - filename: If provided, save the figure to this file
    
    Returns:
    - fig: The matplotlib figure object
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Convert period_start to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(temporal_stats['period_start']):
        temporal_stats['period_start'] = pd.to_datetime(temporal_stats['period_start'])
    
    # Custom date formatting for x-axis
    date_format = mdates.DateFormatter('%Y-%m')
    
    # Plot 1: Number of collaborations over time
    axs[0, 0].plot(temporal_stats['period_start'], temporal_stats['num_collaborations'], marker='o')
    axs[0, 0].set_title('Number of Collaborations Over Time', fontsize=10)
    axs[0, 0].set_ylabel('Collaborations', fontsize=8)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=7)
    axs[0, 0].xaxis.set_major_formatter(date_format)
    
    # Plot 2: Network density over time
    axs[0, 1].plot(temporal_stats['period_start'], temporal_stats['network_density'], marker='o', color='red')
    axs[0, 1].set_title('Network Density Over Time', fontsize=10)
    axs[0, 1].set_ylabel('Density', fontsize=8)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=7)
    axs[0, 1].xaxis.set_major_formatter(date_format)
    
    # Plot 3: Average degree over time
    axs[1, 0].plot(temporal_stats['period_start'], temporal_stats['avg_degree'], marker='o', color='green')
    axs[1, 0].set_title('Average Degree Over Time', fontsize=10)
    axs[1, 0].set_ylabel('Avg. Degree', fontsize=8)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=7)
    axs[1, 0].xaxis.set_major_formatter(date_format)
    
    # Plot 4: Largest component size over time
    axs[1, 1].plot(temporal_stats['period_start'], 
                  temporal_stats['largest_component_size'] / temporal_stats['num_authors'], 
                  marker='o', color='purple')
    axs[1, 1].set_title('Largest Component Size Ratio Over Time', fontsize=10)
    axs[1, 1].set_ylabel('Ratio of Largest Component', fontsize=8)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=7)
    axs[1, 1].xaxis.set_major_formatter(date_format)
    
    # Add more space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    plt.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def plot_field_comparison(field_stats, figsize=(12, 6), filename=None, color_palette="Set2"):
    """
    Plot only the Number of Communities by Field chart.
    
    Parameters:
    - field_stats: DataFrame with field-based collaboration statistics
    - figsize: Figure size as tuple (width, height)
    - filename: If provided, save the figure to this file
    - color_palette: Color palette to use for the bars
    
    Returns:
    - fig: The matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort field_stats by num_communities for better visualization (optional)
    # Uncomment the next line if you want to sort by number of communities
    # field_stats = field_stats.sort_values('num_communities', ascending=False)
    
    # Create the bar plot
    sns.barplot(
        x='field', 
        y='num_communities', 
        data=field_stats, 
        ax=ax, 
        hue='field', 
        palette=color_palette, 
        legend=False
    )
    
    # Customize appearance
    ax.set_title('Number of Communities by Field', fontsize=14, pad=20)
    ax.set_ylabel('Communities', fontsize=12)
    ax.set_xlabel('Field', fontsize=12)
    
    # Handle x-axis labels for better readability
    ax.tick_params(axis='x', rotation=75, labelsize=8)  # Rotated labels with smaller font
    ax.tick_params(axis='y', labelsize=10)
    
    # Add more space at the bottom for rotated labels
    plt.subplots_adjust(bottom=0.3)
    
    # Add grid lines for better readability of values
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout automatically adjusts subplot params for better fit
    plt.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def rebuild_network_visualization(data_file, output_file=None, figsize=(10, 8), edge_width_factor=0.05):
    """
    Rebuild a network visualization from exported raw data without rerunning the full analysis.
    
    Parameters:
    - data_file: Path to the JSON file with network data
    - output_file: Path to save the output figure (optional)
    - figsize: Figure size as tuple (width, height)
    - edge_width_factor: Factor to adjust edge width (higher = thicker edges)
    
    Returns:
    - fig: The matplotlib figure object
    """
    import json
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # Load network data
    with open(data_file, 'r') as f:
        network_data = json.load(f)
    
    # Extract data components
    nodes = network_data['nodes']
    edges = network_data['edges']
    title = network_data.get('title', "Co-authorship Network")
    positions = network_data.get('positions', {})
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in nodes:
        G.add_node(node['id'], degree=node['degree'], field=node['field'], label=node['label'])
    
    # Add edges with weights
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Use saved positions if available, otherwise compute new layout
    if positions:
        pos = {node_id: np.array(pos_coords) for node_id, pos_coords in positions.items()}
    else:
        pos = nx.spring_layout(G, k=1.5)
    
    # Set node sizes based on degree
    node_size = [G.nodes[node]['degree'] * 60 for node in G.nodes()]
    
    # Set edge width (make thinner than default)
    edge_width = [0.1 + (d.get('weight', 1) * edge_width_factor) for u, v, d in G.edges(data=True)]
    
    # Set edge colors (black to gray gradient based on weight)
    max_weight = max([d.get('weight', 1) for u, v, d in G.edges(data=True)]) if G.edges() else 1
    edge_color = [(0.3 + (0.7 * d.get('weight', 1) / max_weight), 
                   0.3 + (0.7 * d.get('weight', 1) / max_weight), 
                   0.3 + (0.7 * d.get('weight', 1) / max_weight)) 
                 for u, v, d in G.edges(data=True)]
    
    # Set node colors based on field if available
    if 'field_colors' in network_data:
        field_colors = network_data['field_colors']
        
        # Convert RGB lists back to tuples for matplotlib
        field_colors = {field: tuple(rgba) for field, rgba in field_colors.items()}
        
        # Map nodes to colors based on their field
        node_colors = [field_colors.get(G.nodes[node]['field'], (0.8, 0.8, 0.8, 1.0)) for node in G.nodes()]
        
        # Create legend patches
        field_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                  markersize=8, label=field) 
                      for field, color in field_colors.items()]
    else:
        # Default node colors if no field information
        node_colors = ['skyblue' for _ in G.nodes()]
        field_patches = []
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.6, edge_color=edge_color)
    
    # Draw labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_family='sans-serif')
    
    # Title and layout
    plt.title(title, fontsize=12)
    plt.axis('off')
    
    # Add field legend if available
    if field_patches:
        plt.legend(handles=field_patches, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    return fig


def main():
    """
    Main function for the academic collaboration analysis script.
    Processes command-line arguments and either runs a full analysis or 
    regenerates visualizations from previously saved results.
    """
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze academic collaboration networks from OpenAlex data')
    parser.add_argument('file_path', type=str, nargs='?', help='Path to the OpenAlex JSON file')
    parser.add_argument('--output_dir', type=str, default='collaboration_analysis_results', 
                        help='Directory to save output files')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a limited sample')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of articles to process in debug mode')
    parser.add_argument('--min_articles', type=int, default=1, 
                        help='Minimum number of articles an author must have to be included')
    parser.add_argument('--min_citations', type=int, default=0,
                        help='Minimum number of citations an article must have to be included')
    parser.add_argument('--regenerate_viz', type=str, 
                        help='Path to previously saved analysis results file to regenerate visualizations')
    
    args = parser.parse_args()
    
    # Check if we're regenerating visualizations from saved results
    if args.regenerate_viz:
        print(f"Regenerating visualizations from saved results: {args.regenerate_viz}")
        
        try:
            with open(args.regenerate_viz, 'rb') as f:
                saved_results = pickle.load(f)
                
            output_dir = args.output_dir or os.path.dirname(args.regenerate_viz)
            regenerate_visualizations(saved_results, output_dir)
            print(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            print(f"Error regenerating visualizations: {e}")
            return 1
            
    else:
        # Ensure we have a file path for full analysis
        if not args.file_path:
            print("Error: You must provide a file path when not using --regenerate_viz")
            parser.print_help()
            return 1
        
        # Run the full analysis
        print(f"Analysis parameters:")
        print(f"  File: {args.file_path}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Debug mode: {'Yes' if args.debug else 'No'}")
        if args.debug:
            print(f"  Sample size: {args.sample_size}")
        print(f"  Minimum articles per author: {args.min_articles}")
        print(f"  Minimum citations per article: {args.min_citations}")
        
        try:
            results = analyze_academic_collaboration(args.file_path, args.output_dir, 
                                                  debug_mode=args.debug,
                                                  sample_size=args.sample_size,
                                                  min_articles_per_author=args.min_articles,
                                                  min_citations=args.min_citations)
            print(f"Analysis complete. Results saved to {args.output_dir}")
            return 0
        except Exception as e:
            print(f"Error running analysis: {e}")
            return 1

# Execute main function when script is run directly
if __name__ == "__main__":
    exit_code = main()# 9. MAIN ANALYSIS FUNCTION