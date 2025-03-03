# OpenAlexAnalysis

A comprehensive analysis framework for understanding and predicting interdisciplinary research collaborations in US academia using OpenAlex data.

## Project Overview

This project analyzes the structure and growth of USA-based scholarly collaborations using OpenAlex data from 2023-2024, focusing on three analytical layers:

1. **Field and Subfield Network Analysis**: Reveals how core disciplines such as Medicine, Biochemistry, and Engineering form prominent hubs, while subfields like Molecular Biology and Biomedical Engineering function as important interdisciplinary bridges.

2. **Author Collaboration Patterns**: Identifies central researchers and evolving communities, showing how fields like Biochemistry and Chemistry form dense collaborative clusters with increasing cross-field linkages.

3. **GNN-based Predictive Modeling**: Implements a temporal Graph Neural Network to predict future interdisciplinary synergies, particularly uncovering emerging connections at the interface of Materials Science and Medicine.

## Repository Structure

```
OpenAlexAnalysis/
├── Authors_Collaboration/         # Analysis of author co-authorship networks
├── Fields_SubFields_Analysis/     # Analysis of field and subfield cooperation networks
├── GNN/                           # Graph Neural Network implementation for prediction
├── download_with_writers.py       # Script for downloading data with writer information
├── downloads.py                   # Core data download utilities
└── run_download.ipynb                      # Jupyter notebook for downloading data
```

## Key Features

- Constructs field and subfield cooperation networks to visualize interdisciplinary connections
- Analyzes co-authorship networks to identify key researchers and community structures
- Employs temporal GNNs to predict future collaboration growth between fields
- Provides rich visualizations of network structures and collaboration patterns
- Identifies emerging interdisciplinary connections with strong growth potential

## Getting Started

### Data Collection

The project uses OpenAlex data from 2023-2024. To download the dataset:

```bash
python downloads.py
```

For downloading data with additional author information:

```bash
python download_with_writers.py
```

### Running the Analysis

You can run each part of the analysis using the following notebooks:

1. **Field & Subfield Network Analysis**: 
   - [Fields_SubFields_Analysis/Fields_SubFields_Grpah.ipynb](Fields_SubFields_Analysis/Fields_SubFields_Grpah.ipynb)

2. **Author Collaboration Analysis**: 
   - [Authors_Collaboration/writer_research.ipynb](Authors_Collaboration/writer_research.ipynb)

3. **GNN Prediction Model**: 
   - [GNN/src/run.ipynb](GNN/src/run.ipynb)

## Results

### Field & Subfield Cooperation Analysis

- Medicine, Biochemistry, and Engineering emerge as top fields by both frequency and network centrality
- About 74.76% of all field connections span different domains, indicating substantial interdisciplinary collaboration
- Molecular Biology emerges as the most connected subfield (194 edges), serving as a key interdisciplinary bridge

### Author Collaboration

- Neuroscience exhibits the highest number of distinct research communities (450+)
- Co-authorship networks form primarily within disciplinary boundaries, with Biochemistry showing strong interconnectedness
- Temporal analysis indicates a steady increase in collaborative work over the study period

### GNN-Based Predictions

- Materials Chemistry (Materials Science) ↔ Oncology (Medicine) emerges as the highest predicted growth pair
- Renewable Energy shows substantial growth when paired with Environmental Science
- Multiple Engineering subfields form strong collaborative bridges with Medicine

## License

This project is licensed under the MIT License - see the LICENSE file for details.
