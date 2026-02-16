"""
Executive Order AI Analysis Package
Analyzes semantic evolution of U.S. executive orders using NLP and ML techniques
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.scraper.federal_register_scraper import FederalRegisterScraper
from src.preprocessing.text_cleaner import TextPreprocessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.analysis.dimensionality_reduction import DimensionalityReducer, ClusterAnalyzer
from src.visualization.plot_generator import PlotGenerator

__all__ = [
    'FederalRegisterScraper',
    'TextPreprocessor',
    'EmbeddingGenerator',
    'DimensionalityReducer',
    'ClusterAnalyzer',
    'PlotGenerator'
]