# Executive Order Analysis

Executive Order Analysis is an end-to-end NLP and machine learning project that examines how U.S. presidential executive order language has evolved over time. The system collects executive orders, converts them into semantic embeddings, identifies latent structure with dimensionality reduction and clustering, and publishes both static and interactive visualizations for exploration.

## Live Website

Explore the published interactive outputs here:

**https://tdl53910.github.io/executive_order_analysis/**

Direct links:
- Interactive time-series plot: https://tdl53910.github.io/executive_order_analysis/interactive_plot.html
- Timeline animation: https://tdl53910.github.io/executive_order_analysis/timeline_animation.html

## What This Project Does

This repository provides a reproducible analysis pipeline for executive orders from recent U.S. administrations. It is designed to answer questions such as:
- How has executive order language shifted semantically over time?
- Do orders cluster into distinct policy/administrative patterns?
- How do presidential eras differ in semantic space?
- Is there measurable year-over-year semantic drift?

The analysis is automated and can be rerun as new executive orders are published.

## Core Capabilities

- **Automated data collection** from the Federal Register API
- **Text preprocessing** for normalization and analysis readiness
- **Sentence embedding generation** using `all-MiniLM-L6-v2`
- **Dimensionality reduction** with PCA, t-SNE, and UMAP
- **Cluster analysis** with K-means and DBSCAN
- **Semantic drift tracking** across years and presidential periods
- **Interactive visual analytics** (HTML/Plotly) and publication-ready static plots

## How the Pipeline Works

1. **Scrape** executive order documents and metadata.
2. **Preprocess** text content.
3. **Embed** each document into a dense semantic vector.
4. **Reduce dimensions** for interpretable structure and plotting.
5. **Cluster** documents to identify broad semantic groupings.
6. **Visualize** trends, trajectories, and cluster composition.
7. **Publish** outputs to GitHub Pages.

The main entry point is:

`scripts/run_analysis.py`

## Configuration and Parameters

Project behavior is controlled through `config.yaml`. Typical parameter groups include:

- **Scraping parameters**
  - `start_year`, `end_year`
  - API query limits and output directories
- **Embedding parameters**
  - model path/name
  - batch size and caching options
- **Reduction parameters**
  - PCA dimensions and variance tracking
  - t-SNE `perplexity` and iteration count
  - UMAP `n_neighbors`, `min_dist`, and random seed
- **Clustering parameters**
  - candidate `k` range for K-means
  - DBSCAN epsilon/min samples
- **Visualization parameters**
  - style, figure size, DPI
  - president color mapping

These settings make it easy to rerun the same workflow for different time windows or tuning assumptions.

## Outputs

Generated artifacts are saved under `output/plots/` and typically include:

- president-based and cluster-based scatter plots
- semantic trend and variance plots
- cluster composition heatmap
- interactive HTML visualizations
- timeline and trajectory views showing temporal change

For web publishing, selected `.html` and `.png` outputs are copied into the Pages publish folder.

## Repository Layout

```text
executive_order_analysis/
├── scripts/              # Pipeline entry points and utilities
├── src/                  # Scraping, preprocessing, embeddings, analysis, visualization
├── output/plots/         # Generated plots and interactive HTML files
├── docs/                 # GitHub Pages publish artifacts
├── config.yaml           # Pipeline configuration
└── README.md
```

## Running Locally

1. Create and activate a virtual environment
2. Install dependencies
3. Run the pipeline

Example:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/run_analysis.py
```

To open generated visualizations locally (macOS):

```bash
open output/plots/interactive_plot.html
open output/plots/timeline_animation.html
```

## Publishing Updated Visualizations

After regenerating outputs, publish refreshed visualizations to GitHub Pages by copying artifacts into the configured publish directory and pushing to GitHub.

## License

MIT License.