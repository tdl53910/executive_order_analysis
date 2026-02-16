#!/usr/bin/env python3
"""
Main script to run the complete executive order analysis pipeline
with local caching to avoid redundant downloads
"""

import sys
from pathlib import Path
import os
import warnings
from dotenv import load_dotenv

# Load environment variables for cache paths
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.federal_register_scraper import FederalRegisterScraper
from src.preprocessing.text_cleaner import TextPreprocessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.analysis.dimensionality_reduction import DimensionalityReducer, ClusterAnalyzer
from src.visualization.plot_generator import PlotGenerator

import logging
import yaml
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_models_downloaded():
    """Check if models are already downloaded"""
    cache_dir = Path(".cache")
    marker = cache_dir / ".download_complete"
    
    if not marker.exists():
        logger.error("="*60)
        logger.error("Models not downloaded yet!")
        logger.error("Please run setup first:")
        logger.error("  ./setup.sh")
        logger.error("  or")
        logger.error("  python scripts/download_models.py")
        logger.error("="*60)
        return False
    
    logger.info("✓ Models found in local cache")
    return True

def run_pipeline():
    """Run the complete analysis pipeline"""
    
    # First check if models are downloaded
    if not check_models_downloaded():
        sys.exit(1)
    
    start_time = datetime.now()
    logger.info("Starting Executive Order Analysis Pipeline")
    logger.info(f"Cache directories:")
    logger.info(f"  - Models: {os.getenv('TRANSFORMERS_CACHE', './models')}")
    logger.info(f"  - Data: ./data")
    logger.info(f"  - Output: ./output")
    
    # Load config to check cache settings
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Scrape data (with caching)
    logger.info("="*50)
    logger.info("STEP 1: Scraping Executive Orders")
    scraper = FederalRegisterScraper()
    
    # Check if we already have data
    raw_files = list(Path(config['data']['raw_dir']).glob('*.txt'))
    if raw_files and config['scraping'].get('use_cache', True):
        logger.info(f"Found {len(raw_files)} existing raw files. Skipping scrape.")
        logger.info("(Delete data/raw/* to force re-scrape)")
    else:
        scraper.scrape_orders()
    
    # Step 2: Preprocess texts (with caching)
    logger.info("="*50)
    logger.info("STEP 2: Preprocessing Texts")
    preprocessor = TextPreprocessor()
    
    # Check if preprocessing is already done
    processed_files = list(Path(config['data']['processed_dir']).glob('*.txt'))
    if processed_files and config['preprocessing'].get('cache_processed', True):
        logger.info(f"Found {len(processed_files)} existing processed files. Skipping preprocessing.")
        logger.info("(Delete data/processed/* to force re-preprocess)")
    else:
        preprocessor.preprocess_all()
    
    # Step 3: Generate embeddings (with caching)
    logger.info("="*50)
    logger.info("STEP 3: Generating Embeddings")
    embedding_gen = EmbeddingGenerator()
    
    # This method already has caching built in
    embeddings, metadata = embedding_gen.generate_all()
    
    if embeddings is None:
        logger.error("Failed to generate embeddings")
        return
    
    # Keep metadata and reduced coordinates aligned for plotting
    if len(metadata) != len(embeddings):
        logger.warning(
            "Metadata length (%d) and embeddings length (%d) differ; truncating both to %d.",
            len(metadata),
            len(embeddings),
            min(len(metadata), len(embeddings))
        )
        n = min(len(metadata), len(embeddings))
        metadata = metadata.iloc[:n].reset_index(drop=True)
        embeddings = embeddings[:n]
    
    # Step 4: Dimensionality reduction (with caching)
    logger.info("="*50)
    logger.info("STEP 4: Dimensionality Reduction")
    reducer = DimensionalityReducer()
    
    # Check if reduction results are cached
    reduced_results = None
    if config['analysis'].get('cache_results', True):
        # Try to load cached results
        try:
            embeddings, metadata = reducer.load_embeddings()
            reduced_results = {}
            for method in ['pca', 'tsne', 'umap']:
                coords_path = Path(config['data']['output_dir']) / f'{method}_coordinates.npy'
                if coords_path.exists():
                    reduced_results[method] = {'reduced': np.load(coords_path)}
            if len(reduced_results) == 3:
                logger.info("Found cached dimensionality reduction results")
        except:
            reduced_results = None
    
    if reduced_results is None:
        reduced_results = reducer.reduce_all(embeddings)
    
    # Step 5: Cluster analysis
    logger.info("="*50)
    logger.info("STEP 5: Cluster Analysis")
    cluster_analyzer = ClusterAnalyzer()
    cluster_results = cluster_analyzer.analyze_clusters(
        embeddings, reduced_results, metadata
    )
    
    # Step 6: Generate visualizations
    logger.info("="*50)
    logger.info("STEP 6: Generating Visualizations")
    plot_gen = PlotGenerator()
    plot_gen.generate_all_plots(reduced_results, cluster_results, metadata)
    
    # Summary
    logger.info("="*50)
    logger.info("PIPELINE COMPLETE")
    elapsed = datetime.now() - start_time
    logger.info(f"Total time: {elapsed}")
    
    # Print cache locations for future reference
    logger.info("\nCache locations:")
    logger.info(f"  - Raw data: {config['data']['raw_dir']}")
    logger.info(f"  - Processed: {config['data']['processed_dir']}")
    logger.info(f"  - Embeddings: {config['data']['embeddings_dir']}")
    logger.info(f"  - Results: {config['data']['output_dir']}")
    
    # Print key findings
    logger.info("\nKey Findings:")
    
    if 'drift' in cluster_results:
        slope = cluster_results['drift']['trend_slope']
        direction = "increasing" if slope > 0 else "decreasing"
        logger.info(f"- Semantic trend: {direction} (slope: {slope:.3f})")
    
    if 'kmeans' in cluster_results:
        n_clusters = cluster_results['kmeans']['n_clusters']
        logger.info(f"- Optimal number of clusters: {n_clusters}")
    
    if 'pca' in reduced_results and 'explained_variance_ratio' in reduced_results['pca']:
        var_95 = reduced_results['pca'].get('n_components_95', 'N/A')
        logger.info(f"- PCA: 95% variance explained by {var_95} components")

if __name__ == "__main__":
    run_pipeline()