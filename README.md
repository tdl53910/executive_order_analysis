# Executive Order Analysis: AI-Powered Semantic Analysis of Presidential Directives

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Live Interactive Results

Your GitHub Pages site is available at:

**https://tdl53910.github.io/executive_order_analysis/**

It may take 2–3 minutes after pushing for updates to appear.

### Direct links
- **Main site**: https://tdl53910.github.io/executive_order_analysis/
- **Interactive plot**: https://tdl53910.github.io/executive_order_analysis/interactive_plot.html
- **Timeline animation**: https://tdl53910.github.io/executive_order_analysis/timeline_animation.html
- **Static image example**: https://tdl53910.github.io/executive_order_analysis/pca_by_president.png

### Check Pages status
1. Open: https://github.com/tdl53910/executive_order_analysis
2. Go to **Settings** → **Pages**
3. Confirm the message: _"Your site is live at https://tdl53910.github.io/executive_order_analysis/"_

## Overview

This project applies NLP and machine learning to analyze the semantic evolution of U.S. executive orders across administrations.

Core pipeline:
- Scrape executive orders from the Federal Register API
- Clean and preprocess text
- Generate sentence embeddings (all-MiniLM-L6-v2)
- Reduce dimensionality (PCA, t-SNE, UMAP)
- Run clustering and drift analysis
- Produce static and interactive visualizations

## Repository Structure

```text
executive_order_analysis/
├── docs/                         # GitHub Pages publish folder
├── output/plots/                 # Generated plots and interactive HTML files
├── scripts/
│   ├── run_analysis.py           # Main pipeline
│   ├── download_models.py
│   └── export_results.py
├── src/
│   ├── scraper/
│   ├── preprocessing/
│   ├── embeddings/
│   ├── analysis/
│   └── visualization/
├── config.yaml
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/tdl53910/executive_order_analysis.git
cd executive_order_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
source venv/bin/activate
python scripts/run_analysis.py
```

## Open local outputs

```bash
open output/plots/interactive_plot.html
open output/plots/timeline_animation.html
open output/plots/president_panels.png
open output/plots/centroid_trajectory.png
```

## Publish latest plots to GitHub Pages

If your Pages source is the repository root (or `/docs`), copy generated files to the publish location and push:

```bash
cp output/plots/*.html docs/
cp output/plots/*.png docs/
git add docs
git commit -m "Update published visualizations"
git push
```

## Notes

- `urllib3` LibreSSL warning on macOS system Python is non-fatal for this workflow.
- If metadata and embeddings differ in length, the pipeline now truncates safely to aligned rows.

## Citation

```text
Eappen, J. & Lent, T. (2026). Executive Order Analysis: AI-Powered Semantic
Analysis of Presidential Directives. GitHub.
https://github.com/tdl53910/executive_order_analysis
```

## License

MIT License (see `LICENSE`).