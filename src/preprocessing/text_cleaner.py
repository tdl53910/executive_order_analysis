# src/preprocessing/text_cleaner.py

import re
import nltk
import spacy
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import logging
from tqdm import tqdm
import yaml
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocess executive order texts for analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spaCy model for lemmatization
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load stopwords
        self.stopwords = set(stopwords.words('english'))
        # Add legal/presidential specific stopwords
        self.stopwords.update([
            'shall', 'hereby', 'pursuant', 'thereof', 'thereto', 
            'herein', 'whereof', 'section', 'subsection', 'paragraph',
            'clause', 'title', 'chapter', 'order', 'executive'
        ])
        
        # Compile regex patterns
        self.patterns = {
            'eo_number': re.compile(r'executive\s+order\s+\d+', re.IGNORECASE),
            'date': re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'),
            'citation': re.compile(r'\[\s*.*?\s*\]'),
            'url': re.compile(r'https?://\S+'),
            'extra_whitespace': re.compile(r'\s+'),
            'non_alpha': re.compile(r'[^a-z\s]')
        }
    
    def preprocess_all(self):
        """Preprocess all files in raw directory"""
        
        # Get all text files
        files = list(self.raw_dir.glob('*.txt'))
        files = [f for f in files if f.name != 'metadata.csv']
        
        logger.info(f"Preprocessing {len(files)} files")
        
        # Load metadata if exists
        metadata = self._load_metadata()
        
        processed_stats = []
        
        for file_path in tqdm(files, desc="Preprocessing"):
            try:
                # Read raw text
                raw_text = file_path.read_text(encoding='utf-8')
                
                # Preprocess
                processed_text, stats = self.preprocess_text(raw_text)
                
                # Save processed text
                output_path = self.processed_dir / file_path.name
                output_path.write_text(processed_text, encoding='utf-8')
                
                # Add metadata if available
                if metadata and file_path.name in metadata:
                    stats.update(metadata[file_path.name])
                
                processed_stats.append(stats)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Save preprocessing statistics
        self._save_stats(processed_stats)
        logger.info(f"Preprocessing complete. Processed {len(processed_stats)} files")
    
    def preprocess_text(self, text: str) -> tuple:
        """Preprocess a single text and return cleaned version with stats"""
        
        # Store original for stats
        original = text
        
        # Remove executive order number and date
        text = self.patterns['eo_number'].sub(' ', text)
        text = self.patterns['date'].sub(' ', text)
        text = self.patterns['citation'].sub(' ', text)
        text = self.patterns['url'].sub(' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-alphabetic characters (keep spaces)
        text = self.patterns['non_alpha'].sub(' ', text)
        
        # Normalize whitespace
        text = self.patterns['extra_whitespace'].sub(' ', text).strip()
        
        # Tokenize and remove stopwords
        if self.config['preprocessing']['remove_stopwords']:
            words = text.split()
            words = [w for w in words if len(w) >= self.config['preprocessing']['min_word_length']]
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)
        
        # Lemmatization (optional, slower)
        if self.config['preprocessing']['use_lemmatization']:
            doc = self.nlp(text[:10000])  # Limit for performance
            text = ' '.join([token.lemma_ for token in doc])
        
        # Calculate statistics
        stats = self._calculate_stats(original, text)
        
        # Truncate if needed
        if len(text) > self.config['preprocessing']['max_text_length']:
            text = text[:self.config['preprocessing']['max_text_length']]
        
        return text, stats
    
    def _calculate_stats(self, original: str, processed: str) -> Dict:
        """Calculate text statistics"""
        
        original_words = len(original.split())
        processed_words = len(processed.split())
        
        # Readability scores
        flesch = textstat.flesch_reading_ease(original)
        kincaid = textstat.flesch_kincaid_grade(original)
        
        # Sentence count
        sentences = sent_tokenize(original)
        
        # Word length distribution
        word_lengths = [len(w) for w in processed.split()]
        
        return {
            'original_words': original_words,
            'processed_words': processed_words,
            'compression_ratio': processed_words / original_words if original_words > 0 else 0,
            'sentence_count': len(sentences),
            'avg_word_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0,
            'flesch_reading_ease': flesch,
            'flesch_kincaid_grade': kincaid,
            'unique_words': len(set(processed.split())) if processed_words > 0 else 0
        }
    
    def _load_metadata(self) -> Dict:
        """Load metadata from CSV file"""
        metadata_path = self.raw_dir / 'metadata.csv'
        if not metadata_path.exists():
            return {}
        
        df = pd.read_csv(metadata_path)
        metadata = {}
        for _, row in df.iterrows():
            metadata[row['filename']] = {
                'president': row.get('president', 'Unknown'),
                'date': row.get('date', ''),
                'eo_number': row.get('eo_number', '')
            }
        return metadata
    
    def _save_stats(self, stats: List[Dict]):
        """Save preprocessing statistics"""
        if not stats:
            return
        
        df = pd.DataFrame(stats)
        stats_path = self.processed_dir / 'preprocessing_stats.csv'
        df.to_csv(stats_path, index=False)
        logger.info(f"Statistics saved to {stats_path}")