# src/analysis/dimensionality_reduction.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
import yaml
import joblib
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import inspect

logger = logging.getLogger(__name__)

class DimensionalityReducer:
    """Perform dimensionality reduction on embeddings"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Normalize analysis/tsne config with safe defaults
        analysis_cfg = self.config.get('analysis', {})
        tsne_cfg = analysis_cfg.get('tsne', self.config.get('tsne', {})) or {}
        tsne_cfg = dict(tsne_cfg) if tsne_cfg is not None else {}
        tsne_cfg.setdefault('perplexity', 30)
        tsne_cfg.setdefault('max_iter', tsne_cfg.get('n_iter', 1000))
        tsne_cfg.setdefault('n_iter', tsne_cfg.get('max_iter', 1000))
        self.config['tsne'] = tsne_cfg

        self.embeddings_dir = Path(self.config['data']['embeddings_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pca_components = self.config['analysis']['pca_components']
        self.tsne_iterations = tsne_cfg.get('n_iter')
        self.tsne_perplexity = tsne_cfg.get('perplexity')
        self.umap_neighbors = self.config['analysis']['umap_neighbors']
        self.umap_min_dist = self.config['analysis']['umap_min_dist']
        
    def load_embeddings(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load embeddings and metadata"""
        
        embeddings_path = self.embeddings_dir / 'embeddings.npy'
        metadata_path = self.embeddings_dir / 'embedding_metadata.csv'
        
        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Embeddings or metadata not found")
        
        embeddings = np.load(embeddings_path)
        metadata = pd.read_csv(metadata_path)
        
        logger.info(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        return embeddings, metadata
    
    def reduce_all(self, embeddings: np.ndarray) -> Dict:
        """Apply all dimensionality reduction techniques"""
        
        results = {}
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # PCA
        logger.info("Performing PCA...")
        pca_results = self._apply_pca(embeddings_scaled)
        results['pca'] = pca_results
        
        # t-SNE (use PCA-reduced for speed if embeddings are high-dimensional)
        logger.info("Performing t-SNE...")
        tsne_input = pca_results['reduced'] if embeddings.shape[1] > 50 else embeddings_scaled
        tsne_results = self._apply_tsne(tsne_input)
        results['tsne'] = tsne_results
        
        # UMAP
        logger.info("Performing UMAP...")
        umap_results = self._apply_umap(embeddings_scaled)
        results['umap'] = umap_results
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _apply_pca(self, embeddings: np.ndarray) -> Dict:
        """Apply PCA dimensionality reduction"""
        
        pca = PCA(n_components=min(self.pca_components, embeddings.shape[1]))
        reduced = pca.fit_transform(embeddings)
        
        # Calculate cumulative explained variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.searchsorted(cumsum, 0.95) + 1
        
        results = {
            'reduced': reduced,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumsum,
            'n_components_95': n_components_95,
            'components': pca.components_,
            'model': pca
        }
        
        logger.info(f"PCA: {reduced.shape[1]} dimensions, "
                   f"{cumsum[-1]:.2%} variance explained")
        logger.info(f"95% variance requires {n_components_95} components")
        
        return results
    
    def _apply_tsne(self, embeddings: np.ndarray) -> Dict:
        """Apply t-SNE dimensionality reduction"""

        analysis_cfg = self.config.get('analysis', {})
        tsne_cfg = analysis_cfg.get('tsne', self.config.get('tsne', {})) or {}
        perplexity = getattr(self, 'tsne_perplexity', tsne_cfg.get('perplexity', 30))
        n_iter = getattr(self, 'tsne_iterations', tsne_cfg.get('n_iter', tsne_cfg.get('max_iter', 1000)))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=self.config.get('random_state', 42)
        )

        reduced = tsne.fit_transform(embeddings)

        results = {
            'reduced': reduced,
            'model': tsne
        }

        logger.info("t-SNE complete")

        return results
    
    def _apply_umap(self, embeddings: np.ndarray) -> Dict:
        """Apply UMAP dimensionality reduction"""
        
        reducer = umap.UMAP(
            n_neighbors=self.umap_neighbors,
            min_dist=self.umap_min_dist,
            n_components=2,
            random_state=42,
            n_jobs=-1
        )
        
        reduced = reducer.fit_transform(embeddings)
        
        results = {
            'reduced': reduced,
            'model': reducer
        }
        
        logger.info(f"UMAP complete")
        
        return results
    
    def _save_results(self, results: Dict):
        """Save reduction results"""
        
        # Save reduced coordinates
        for method, data in results.items():
            reduced_path = self.output_dir / f'{method}_coordinates.npy'
            np.save(reduced_path, data['reduced'])
            
            # Save model
            model_path = self.output_dir / f'{method}_model.pkl'
            joblib.dump(data['model'], model_path)
        
        # Save PCA variance info
        if 'pca' in results:
            variance_df = pd.DataFrame({
                'component': range(1, len(results['pca']['explained_variance_ratio']) + 1),
                'explained_variance': results['pca']['explained_variance_ratio'],
                'cumulative_variance': results['pca']['cumulative_variance']
            })
            variance_df.to_csv(self.output_dir / 'pca_variance.csv', index=False)
        
        logger.info(f"Results saved to {self.output_dir}")


class ClusterAnalyzer:
    """Analyze clusters in embedding space"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_clusters = self.config['analysis']['n_clusters']
        self.output_dir = Path(self.config['data']['output_dir'])
        
    def analyze_clusters(self, embeddings: np.ndarray, 
                        reduced_coords: Dict, 
                        metadata: pd.DataFrame) -> Dict:
        """Perform cluster analysis"""
        
        results = {}
        
        # K-means clustering
        logger.info("Performing K-means clustering...")
        kmeans_results = self._kmeans_clustering(embeddings)
        results['kmeans'] = kmeans_results
        
        # DBSCAN clustering
        logger.info("Performing DBSCAN clustering...")
        dbscan_results = self._dbscan_clustering(embeddings)
        results['dbscan'] = dbscan_results
        
        # Analyze clusters by president and year
        logger.info("Analyzing cluster composition...")
        composition = self._analyze_composition(
            kmeans_results['labels'], 
            metadata
        )
        results['composition'] = composition
        
        # Calculate semantic drift
        logger.info("Calculating semantic drift...")
        drift = self._calculate_semantic_drift(embeddings, metadata)
        results['drift'] = drift
        
        # Save results
        self._save_cluster_results(results, reduced_coords)
        
        return results
    
    def _kmeans_clustering(self, embeddings: np.ndarray) -> Dict:
        """Apply K-means clustering"""
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        K_range = range(2, min(10, len(embeddings) // 10))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
        
        # Use optimal k or default
        optimal_k = K_range[np.argmax(silhouette_scores)] if silhouette_scores else self.n_clusters
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        results = {
            'labels': labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_clusters': optimal_k,
            'silhouette_scores': silhouette_scores,
            'model': kmeans
        }
        
        return results
    
    def _dbscan_clustering(self, embeddings: np.ndarray) -> Dict:
        """Apply DBSCAN clustering"""
        
        # Try different epsilon values
        eps_values = np.linspace(0.1, 2.0, 10)
        best_score = -1
        best_labels = None
        best_eps = None
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
            labels = dbscan.fit_predict(embeddings)
            
            # Skip if all noise
            if len(set(labels)) <= 1:
                continue
            
            # Calculate silhouette score (excluding noise points)
            mask = labels != -1
            if mask.sum() > 1:
                try:
                    score = silhouette_score(embeddings[mask], labels[mask])
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_eps = eps
                except:
                    pass
        
        results = {
            'labels': best_labels if best_labels is not None else np.zeros(len(embeddings)),
            'eps': best_eps,
            'silhouette_score': best_score,
            'n_clusters': len(set(best_labels)) - (1 if -1 in best_labels else 0) if best_labels is not None else 0,
            'n_noise': sum(best_labels == -1) if best_labels is not None else 0
        }
        
        return results
    
    def _analyze_composition(self, labels: np.ndarray, metadata: pd.DataFrame) -> Dict:
        """Analyze cluster composition by president and year"""
        
        composition = {
            'by_president': {},
            'by_year': {},
            'cluster_sizes': {}
        }
        
        # Overall cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        composition['cluster_sizes'] = dict(zip(unique, counts))
        
        # Composition by president
        for cluster in unique:
            mask = labels == cluster
            presidents = metadata.loc[mask, 'president']
            pres_counts = presidents.value_counts()
            composition['by_president'][cluster] = pres_counts.to_dict()
        
        # Composition by year
        for cluster in unique:
            mask = labels == cluster
            years = metadata.loc[mask, 'year']
            year_counts = years[years != 'Unknown'].value_counts().sort_index()
            composition['by_year'][cluster] = year_counts.to_dict()
        
        return composition
    
    def _calculate_semantic_drift(self, embeddings: np.ndarray, 
                                 metadata: pd.DataFrame) -> Dict:
        """Calculate semantic drift over time"""
        
        # Group by year
        metadata['year'] = pd.to_numeric(metadata['year'], errors='coerce')
        valid_mask = metadata['year'].notna()
        
        if not valid_mask.any():
            return {'error': 'No valid years found'}
        
        years = metadata.loc[valid_mask, 'year'].astype(int)
        year_embeddings = embeddings[valid_mask]
        
        # Calculate centroid for each year
        unique_years = sorted(years.unique())
        centroids = []
        
        for year in unique_years:
            year_mask = years == year
            if year_mask.sum() > 0:
                centroid = year_embeddings[year_mask].mean(axis=0)
                centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Calculate drift between consecutive years
        drifts = []
        for i in range(1, len(centroids)):
            # Cosine distance
            dot_product = np.dot(centroids[i], centroids[i-1])
            norm_product = np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[i-1])
            cosine_sim = dot_product / norm_product if norm_product > 0 else 0
            drift = 1 - cosine_sim
            drifts.append({
                'from_year': unique_years[i-1],
                'to_year': unique_years[i],
                'drift': drift
            })
        
        # Calculate overall trend
        if len(unique_years) > 1:
            # Linear regression on centroid positions
            from sklearn.linear_model import LinearRegression
            
            # Use first PCA component as proxy for position
            pca = PCA(n_components=1)
            positions = pca.fit_transform(centroids).flatten()
            
            X = np.array(range(len(unique_years))).reshape(-1, 1)
            reg = LinearRegression().fit(X, positions)
            trend_slope = reg.coef_[0]
        else:
            trend_slope = 0
        
        results = {
            'yearly_centroids': centroids,
            'yearly_drifts': drifts,
            'trend_slope': trend_slope,
            'years': unique_years
        }
        
        return results
    
    def _save_cluster_results(self, results: Dict, reduced_coords: Dict):
        """Save cluster analysis results"""
        
        # Save labels and composition
        for method in ['kmeans', 'dbscan']:
            if method in results:
                labels_path = self.output_dir / f'{method}_labels.npy'
                np.save(labels_path, results[method]['labels'])
        
        # Save composition as CSV
        if 'composition' in results:
            # Flatten for CSV
            rows = []
            for cluster, presidents in results['composition']['by_president'].items():
                for president, count in presidents.items():
                    rows.append({
                        'cluster': cluster,
                        'president': president,
                        'count': count
                    })
            
            comp_df = pd.DataFrame(rows)
            comp_df.to_csv(self.output_dir / 'cluster_composition.csv', index=False)
        
        # Save drift results
        if 'drift' in results and 'yearly_drifts' in results['drift']:
            drift_df = pd.DataFrame(results['drift']['yearly_drifts'])
            drift_df.to_csv(self.output_dir / 'semantic_drift.csv', index=False)
        
        logger.info(f"Cluster results saved to {self.output_dir}")
