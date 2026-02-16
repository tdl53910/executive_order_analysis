import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import logging
from tqdm import tqdm
import yaml
import joblib
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.embeddings_dir = Path(self.config['data']['embeddings_dir'])
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir = Path(self.config['embeddings'].get('model_cache_dir', 'models'))
        self.model_name = self.config['embeddings']['model_name']
        self.model = self._load_local_model()
        self.batch_size = self.config['embeddings']['batch_size']
        self.max_length = self.config['embeddings']['max_length']
        self.cache_embeddings = self.config['embeddings'].get('cache_embeddings', True)
        
    def _load_local_model(self) -> SentenceTransformer:
        possible_paths = [
            self.model_cache_dir / 'sentence_transformers' / self.model_name.replace('/', '_'),
            Path.home() / '.cache' / 'torch' / 'sentence_transformers' / self.model_name.replace('/', '_'),
            Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--{self.model_name.replace("/", "--")}'
        ]
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                logger.info(f"Found cached model at: {model_path}")
                break
        if model_path:
            logger.info(f"Loading model from local cache: {model_path}")
            model = SentenceTransformer(str(model_path))
        else:
            logger.warning("Model not found in cache. Downloading...")
            model = SentenceTransformer(self.model_name)
            save_path = self.model_cache_dir / 'sentence_transformers' / self.model_name.replace('/', '_')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
        device = 'cuda' if (self.config['embeddings'].get('use_gpu', False) and torch.cuda.is_available()) else 'cpu'
        if device != model.device.type:
            model = model.to(device)
        return model
    
    def _get_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _check_cached_embeddings(self, files: List[Path]) -> Tuple[bool, Optional[np.ndarray], Optional[List[str]]]:
        embeddings_path = self.embeddings_dir / 'embeddings.npy'
        names_path = self.embeddings_dir / 'file_names.txt'
        hash_path = self.embeddings_dir / 'file_hashes.json'
        if not (embeddings_path.exists() and names_path.exists() and hash_path.exists()):
            return False, None, None
        with open(hash_path, 'r') as f:
            cached_hashes = json.load(f)
        current_hashes = {}
        for file_path in files:
            current_hashes[file_path.name] = self._get_file_hash(file_path)
        if current_hashes == cached_hashes:
            logger.info("Found valid cached embeddings")
            embeddings = np.load(embeddings_path)
            with open(names_path, 'r') as f:
                file_names = [line.strip() for line in f]
            return True, embeddings, file_names
        return False, None, None
    
    def generate_all(self, force_recompute: bool = False):
        files = list(self.processed_dir.glob('*.txt'))
        files = [f for f in files if f.name != 'preprocessing_stats.csv']
        if not files:
            logger.error("No processed files found")
            return None, None
        if not force_recompute and self.cache_embeddings:
            cache_valid, embeddings, file_names = self._check_cached_embeddings(files)
            if cache_valid:
                metadata = self._load_metadata(file_names)
                return embeddings, metadata
        logger.info(f"Generating embeddings for {len(files)} files")
        texts = []
        file_names = []
        file_hashes = {}
        for file_path in tqdm(files, desc="Loading texts"):
            try:
                text = file_path.read_text(encoding='utf-8')
                if text and len(text) > 50:
                    texts.append(text)
                    file_names.append(file_path.name)
                    file_hashes[file_path.name] = self._get_file_hash(file_path)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        if not texts:
            logger.error("No valid texts found")
            return None, None
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            batch_texts = [t[:self.max_length * 4] for t in batch_texts]
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
        embeddings = np.vstack(all_embeddings)
        if self.cache_embeddings:
            self._save_embeddings(embeddings, file_names, file_hashes)
        metadata = self._load_metadata(file_names)
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        return embeddings, metadata
    
    def _save_embeddings(self, embeddings: np.ndarray, file_names: List[str], file_hashes: Dict):
        np.save(self.embeddings_dir / 'embeddings.npy', embeddings)
        with open(self.embeddings_dir / 'file_names.txt', 'w') as f:
            f.write('\n'.join(file_names))
        with open(self.embeddings_dir / 'file_hashes.json', 'w') as f:
            json.dump(file_hashes, f)
        logger.info(f"Embeddings cached")
    
    def _load_metadata(self, file_names: List[str]) -> Optional[pd.DataFrame]:
        metadata_path = self.processed_dir / 'preprocessing_stats.csv'
        metadata_list = []
        for fname in file_names:
            parts = fname.replace('EO_', '').replace('.txt', '').split('_')
            eo_num = parts[0] if len(parts) > 0 else ''
            date_str = parts[1] if len(parts) > 1 else ''
            year = date_str[:4] if date_str and len(date_str) >= 4 else 'Unknown'
            president = 'Unknown'
            if year and year.isdigit():
                year_int = int(year)
                if 1993 <= year_int <= 2001:
                    president = 'Clinton'
                elif 2001 <= year_int <= 2009:
                    president = 'Bush'
                elif 2009 <= year_int <= 2017:
                    president = 'Obama'
                elif 2017 <= year_int <= 2021:
                    president = 'Trump'
                elif 2021 <= year_int <= 2025:
                    president = 'Biden'
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                if eo_num and 'eo_number' in df.columns:
                    match = df[df['eo_number'].astype(str) == eo_num]
                    if len(match) > 0:
                        president = match.iloc[0].get('president', president)
            metadata_list.append({
                'filename': fname,
                'eo_number': eo_num,
                'date': date_str,
                'year': year,
                'president': president
            })
        return pd.DataFrame(metadata_list)

