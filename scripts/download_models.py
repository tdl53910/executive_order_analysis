#!/usr/bin/env python3
"""
Script to download all required models ONCE and cache them locally.
Run this once before running the analysis.
"""

import os
import logging
import nltk
import spacy
from sentence_transformers import SentenceTransformer
import torch
import yaml
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Download and cache all required models locally"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up cache directories
        self.cache_dir = Path(self.config['data'].get('cache_dir', '.cache'))
        self.model_cache_dir = Path(self.config['embeddings'].get('model_cache_dir', 'models'))
        
        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for caching
        os.environ['TRANSFORMERS_CACHE'] = str(self.model_cache_dir / 'transformers')
        os.environ['HF_HOME'] = str(self.model_cache_dir / 'huggingface')
        os.environ['TORCH_HOME'] = str(self.model_cache_dir / 'torch')
        
        # NLTK data path
        nltk.data.path.append(str(self.cache_dir / 'nltk_data'))
        
    def download_all(self):
        """Download all models and data"""
        
        logger.info("="*50)
        logger.info("Downloading all required models (ONE-TIME SETUP)")
        logger.info("="*50)
        
        # 1. Download NLTK data
        logger.info("\n1. Downloading NLTK data...")
        nltk_data_dir = self.cache_dir / 'nltk_data'
        nltk_data_dir.mkdir(exist_ok=True)
        
        nltk.download('punkt', download_dir=str(nltk_data_dir), quiet=False)
        nltk.download('stopwords', download_dir=str(nltk_data_dir), quiet=False)
        nltk.download('averaged_perceptron_tagger', download_dir=str(nltk_data_dir), quiet=False)
        nltk.download('wordnet', download_dir=str(nltk_data_dir), quiet=False)
        
        logger.info(f"✓ NLTK data downloaded to {nltk_data_dir}")
        
        # 2. Download spaCy model
        logger.info("\n2. Downloading spaCy model...")
        try:
            # Check if already downloaded
            spacy.load('en_core_web_sm')
            logger.info("✓ spaCy model already downloaded")
        except:
            logger.info("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            # Download to custom location
            os.system(f"python -m spacy download en_core_web_sm")
        
        # 3. Download sentence transformer model
        logger.info("\n3. Downloading sentence transformer model...")
        model_name = self.config['embeddings']['model_name']
        model_path = self.model_cache_dir / 'sentence_transformers' / model_name.replace('/', '_')
        
        if model_path.exists():
            logger.info(f"✓ Model already cached at {model_path}")
            # Test load
            model = SentenceTransformer(str(model_path))
        else:
            logger.info(f"Downloading {model_name} (this may take a few minutes)...")
            # Download and save locally
            model = SentenceTransformer(model_name)
            save_path = self.model_cache_dir / 'sentence_transformers' / model_name.replace('/', '_')
            model.save(str(save_path))
            logger.info(f"✓ Model saved to {save_path}")
        
        # Test with a simple sentence
        test_embedding = model.encode("This is a test sentence.")
        logger.info(f"  Model test successful. Embedding dimension: {len(test_embedding)}")
        
        # 4. Check GPU availability
        logger.info("\n4. Checking hardware...")
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
        else:
            logger.info("ℹ GPU not available, using CPU")
        
        # 5. Create a marker file to indicate download complete
        marker_file = self.cache_dir / '.download_complete'
        with open(marker_file, 'w') as f:
            f.write(f"Download completed on: {pd.Timestamp.now()}\n")
            f.write(f"Model: {model_name}\n")
        
        logger.info("\n" + "="*50)
        logger.info("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info("\nYou can now run the analysis without downloading again.")
        logger.info("Run: python scripts/run_analysis.py")
        
    def check_download_status(self):
        """Check if models are already downloaded"""
        
        marker_file = self.cache_dir / '.download_complete'
        
        if marker_file.exists():
            logger.info("✓ Models already downloaded")
            with open(marker_file, 'r') as f:
                content = f.read()
                logger.info(f"  {content.strip()}")
            return True
        else:
            logger.info("Models not downloaded yet")
            return False

if __name__ == "__main__":
    import pandas as pd
    downloader = ModelDownloader()
    
    # Check if already downloaded
    if not downloader.check_download_status():
        response = input("Models not found. Download now? (y/n): ")
        if response.lower() == 'y':
            downloader.download_all()
        else:
            print("Exiting. Run this script again when ready to download.")
    else:
        response = input("Models already downloaded. Download again? (y/n): ")
        if response.lower() == 'y':
            downloader.download_all()