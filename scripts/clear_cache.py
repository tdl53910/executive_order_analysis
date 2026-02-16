#!/usr/bin/env python3
# scripts/clear_cache.py - Utility to clear caches selectively

import shutil
from pathlib import Path
import argparse
import yaml

def clear_cache(cache_type: str, config_path: str = "config.yaml"):
    """Clear specified cache"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cache_paths = {
        'raw': Path(config['data']['raw_dir']),
        'processed': Path(config['data']['processed_dir']),
        'embeddings': Path(config['data']['embeddings_dir']),
        'output': Path(config['data']['output_dir']),
        'models': Path(config['embeddings'].get('model_cache_dir', 'models')),
        'cache': Path(config['data'].get('cache_dir', '.cache')),
        'all': None
    }
    
    if cache_type == 'all':
        for name, path in cache_paths.items():
            if name != 'all' and path and path.exists():
                print(f"Clearing {name} cache: {path}")
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
    else:
        path = cache_paths.get(cache_type)
        if path and path.exists():
            print(f"Clearing {cache_type} cache: {path}")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Cache {cache_type} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear caches')
    parser.add_argument('type', choices=['raw', 'processed', 'embeddings', 
                                        'output', 'models', 'cache', 'all'],
                       help='Type of cache to clear')
    args = parser.parse_args()
    
    clear_cache(args.type)