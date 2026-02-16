# src/visualization/plot_generator.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
from typing import Dict, List, Optional
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class PlotGenerator:
    """Generate visualizations for executive order analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.config['visualization']['style'])
        self.figsize = tuple(self.config['visualization']['figure_size'])
        self.dpi = self.config['visualization']['dpi']
        self.president_colors = self.config['visualization']['president_colors']
        
        # President markers for scatter plots
        self.president_markers = {
            'Clinton': 'o',
            'Bush': 's',
            'Obama': '^',
            'Trump': 'D',
            'Biden': 'X'
        }
    
    def generate_all_plots(self, reduced_results: Dict, 
                          cluster_results: Dict,
                          metadata: pd.DataFrame):
        """Generate all visualization plots"""
        
        logger.info("Generating visualizations...")
        
        pca_coords = reduced_results['pca']['reduced']
        
        # Scatter plots by president
        self.plot_president_scatter(
            pca_coords,
            metadata,
            'PCA',
            'pca_by_president.png'
        )
        
        self.plot_president_scatter(
            reduced_results['tsne']['reduced'],
            metadata,
            't-SNE',
            'tsne_by_president.png'
        )
        
        self.plot_president_scatter(
            reduced_results['umap']['reduced'],
            metadata,
            'UMAP',
            'umap_by_president.png'
        )
        
        # Cluster plots
        if 'kmeans' in cluster_results:
            self.plot_clusters(
                reduced_results['pca']['reduced'],
                cluster_results['kmeans']['labels'],
                'PCA',
                'pca_clusters.png'
            )
        
        # Time trend plots
        if 'drift' in cluster_results:
            self.plot_semantic_trend(cluster_results['drift'])
        
        # PCA variance plot
        if 'pca' in reduced_results and 'explained_variance_ratio' in reduced_results['pca']:
            self.plot_pca_variance(reduced_results['pca'])
        
        # Heatmap of cluster composition
        if 'composition' in cluster_results:
            self.plot_cluster_composition(cluster_results['composition'])
        
        # Interactive plot (HTML)
        self.plot_interactive(
            pca_coords,
            metadata,
            cluster_results.get('kmeans', {}).get('labels', None),
            'interactive_plot.html'
        )

        # Time-based plots
        self.plot_animated_timeline(
            pca_coords,
            metadata,
            'timeline_animation.html'
        )

        self.plot_by_president_panel(
            pca_coords,
            metadata,
            'president_panels.png'
        )

        self.plot_centroid_trajectory(
            pca_coords,
            metadata,
            'centroid_trajectory.png'
        )
        
        logger.info(f"All plots saved to {self.output_dir}")
    
    def plot_president_scatter(self, coordinates: np.ndarray, 
                              metadata: pd.DataFrame,
                              method: str, 
                              filename: str):
        """Create scatter plot colored by president"""
        
        plt.figure(figsize=self.figsize)
        
        # Plot points for each president
        for president in metadata['president'].unique():
            if president == 'Unknown':
                continue
                
            mask = metadata['president'] == president
            if mask.sum() == 0:
                continue
                
            plt.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=self.president_colors.get(president, '#333333'),
                marker=self.president_markers.get(president, 'o'),
                label=president,
                alpha=0.7,
                s=50
            )
        
        plt.title(f'Executive Orders - {method} Projection by President')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {filename}")
    
    def plot_clusters(self, coordinates: np.ndarray, 
                     labels: np.ndarray,
                     method: str,
                     filename: str):
        """Create scatter plot with clusters highlighted"""
        
        plt.figure(figsize=self.figsize)
        
        # Use a colormap for clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=[colors[i]],
                label=f'Cluster {label}',
                alpha=0.7,
                s=50
            )
        
        plt.title(f'Executive Orders - {method} with K-means Clusters')
        plt.xlabel(f'{method} Component 1')
        plt.ylabel(f'{method} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {filename}")
    
    def plot_semantic_trend(self, drift_results: Dict):
        """Plot semantic drift over time"""
        
        years = drift_results['years']
        drifts = drift_results['yearly_drifts']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        # Plot 1: Drift between consecutive years
        drift_years = [f"{d['from_year']}-{d['to_year']}" for d in drifts]
        drift_values = [d['drift'] for d in drifts]
        
        ax1.bar(drift_years, drift_values, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Year Transition')
        ax1.set_ylabel('Semantic Drift')
        ax1.set_title('Semantic Drift Between Consecutive Years')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Centroid positions over time (using PCA of centroids)
        if len(years) > 1:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            positions = pca.fit_transform(drift_results['yearly_centroids']).flatten()
            
            ax2.plot(years, positions, 'o-', color='darkred', linewidth=2, markersize=8)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Position (PC1)')
            ax2.set_title('Semantic Center of Gravity Over Time')
            
            # Add trend line
            z = np.polyfit(range(len(years)), positions, 1)
            p = np.poly1d(z)
            ax2.plot(years, p(range(len(years))), '--', color='gray', 
                    label=f'Trend (slope: {z[0]:.3f})')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'semantic_trend.png', dpi=self.dpi)
        plt.close()
        
        logger.info("Saved semantic_trend.png")
    
    def plot_pca_variance(self, pca_results: Dict):
        """Plot PCA explained variance"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]))
        
        # Individual explained variance
        components = range(1, len(pca_results['explained_variance_ratio']) + 1)
        ax1.bar(components[:20], pca_results['explained_variance_ratio'][:20], 
                alpha=0.7, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Explained Variance')
        
        # Cumulative variance
        ax2.plot(components, pca_results['cumulative_variance'] * 100, 
                'o-', color='darkred', linewidth=2, markersize=4)
        ax2.axhline(y=95, color='gray', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance (%)')
        ax2.set_title('Cumulative Explained Variance')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_variance.png', dpi=self.dpi)
        plt.close()
        
        logger.info("Saved pca_variance.png")
    
    def plot_cluster_composition(self, composition: Dict):
        """Plot cluster composition by president"""

        # Create heatmap data
        presidents = list(self.president_colors.keys())
        clusters = sorted(composition['by_president'].keys())

        heatmap_data = []
        for cluster in clusters:
            row = []
            for president in presidents:
                row.append(composition['by_president'][cluster].get(president, 0))
            heatmap_data.append(row)

        # Convert to float array before division
        heatmap_data = np.array(heatmap_data, dtype=float)
        row_sums = heatmap_data.sum(axis=1, keepdims=True)

        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0

        # Normalize
        heatmap_data = heatmap_data / row_sums

        # Plot
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, 
                   xticklabels=presidents,
                   yticklabels=[f'Cluster {c}' for c in clusters],
                   annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Proportion'})

        plt.title('Cluster Composition by President')
        plt.xlabel('President')
        plt.ylabel('Cluster')
        plt.tight_layout()

        plt.savefig(self.output_dir / 'cluster_composition.png', dpi=self.dpi)
        plt.close()

        logger.info("Saved cluster_composition.png")
    
    def plot_interactive(self, coordinates: np.ndarray, 
                        metadata: pd.DataFrame,
                        cluster_labels: Optional[np.ndarray],
                        filename: str):
        """Create interactive time-series plot (x-axis = year)"""

        n_points = min(len(metadata), len(coordinates))
        if n_points == 0:
            logger.warning("No points available for interactive plot.")
            return

        plot_df = metadata.iloc[:n_points].copy().reset_index(drop=True)
        coords = coordinates[:n_points]

        plot_df['pc1'] = coords[:, 0]
        plot_df['pc2'] = coords[:, 1]
        plot_df['year_num'] = pd.to_numeric(plot_df['year'], errors='coerce')

        if cluster_labels is not None:
            cluster_labels = np.asarray(cluster_labels)[:n_points]
            plot_df['cluster'] = cluster_labels
        else:
            plot_df['cluster'] = None

        plot_df = plot_df.dropna(subset=['year_num']).copy()
        if plot_df.empty:
            logger.warning("No valid year values found for interactive plot.")
            return

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Semantic Position Over Time (PC1)',
                'Semantic Position Over Time (PC2)'
            )
        )

        presidents = [p for p in plot_df['president'].dropna().unique()]
        for president in presidents:
            df_p = plot_df[plot_df['president'] == president]
            if df_p.empty:
                continue

            hover_text = []
            for _, row in df_p.iterrows():
                text = f"File: {row.get('filename', 'N/A')}<br>"
                text += f"President: {row.get('president', 'Unknown')}<br>"
                text += f"Year: {int(row['year_num']) if pd.notna(row['year_num']) else 'N/A'}<br>"
                text += f"PC1: {row['pc1']:.3f}<br>PC2: {row['pc2']:.3f}"
                if row.get('cluster') is not None:
                    text += f"<br>Cluster: {row['cluster']}"
                hover_text.append(text)

            fig.add_trace(
                go.Scatter(
                    x=df_p['year_num'],
                    y=df_p['pc1'],
                    mode='markers',
                    name=president,
                    legendgroup=president,
                    showlegend=True,
                    marker=dict(
                        color=self.president_colors.get(president, '#333333'),
                        size=7,
                        opacity=0.75
                    ),
                    text=hover_text,
                    hoverinfo='text'
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_p['year_num'],
                    y=df_p['pc2'],
                    mode='markers',
                    name=president,
                    legendgroup=president,
                    showlegend=False,
                    marker=dict(
                        color=self.president_colors.get(president, '#333333'),
                        size=7,
                        opacity=0.75
                    ),
                    text=hover_text,
                    hoverinfo='text'
                ),
                row=2,
                col=1
            )

        if len(plot_df) > 1:
            x_sorted = np.sort(plot_df['year_num'].values)

            z1 = np.polyfit(plot_df['year_num'], plot_df['pc1'], 1)
            p1 = np.poly1d(z1)
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=p1(x_sorted),
                    mode='lines',
                    name=f"PC1 Trend (slope={z1[0]:.3f})",
                    line=dict(color='black', width=2, dash='dash')
                ),
                row=1,
                col=1
            )

            z2 = np.polyfit(plot_df['year_num'], plot_df['pc2'], 1)
            p2 = np.poly1d(z2)
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=p2(x_sorted),
                    mode='lines',
                    name=f"PC2 Trend (slope={z2[0]:.3f})",
                    line=dict(color='gray', width=2, dash='dash')
                ),
                row=2,
                col=1
            )

        fig.update_xaxes(title_text='Year', row=2, col=1)
        fig.update_yaxes(title_text='PC1', row=1, col=1)
        fig.update_yaxes(title_text='PC2', row=2, col=1)

        fig.update_layout(
            title='Executive Orders - Semantic Time Series (x-axis = Year)',
            hovermode='closest',
            width=1100,
            height=800,
            legend_title_text='President'
        )

        fig.write_html(self.output_dir / filename)
        logger.info(f"Saved {filename}")

    def plot_animated_timeline(self, coordinates: np.ndarray,
                               metadata: pd.DataFrame,
                               filename: str = 'timeline_animation.html'):
        """Create animated year-by-year semantic evolution plot"""

        n_points = min(len(metadata), len(coordinates))
        if n_points == 0:
            logger.warning("No points available for animated timeline.")
            return

        plot_df = metadata.iloc[:n_points].copy().reset_index(drop=True)
        coords = coordinates[:n_points]
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        plot_df['year_num'] = pd.to_numeric(plot_df['year'], errors='coerce')
        plot_df = plot_df.dropna(subset=['year_num']).copy()
        if plot_df.empty:
            logger.warning("No valid year values found for animated timeline.")
            return

        plot_df['year_num'] = plot_df['year_num'].astype(int)

        hover_cols = [c for c in ['filename', 'eo_number', 'president'] if c in plot_df.columns]

        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            animation_frame='year_num',
            color='president' if 'president' in plot_df.columns else None,
            hover_data=hover_cols,
            title='Executive Orders Evolution Over Time',
            labels={'x': 'Component 1', 'y': 'Component 2', 'year_num': 'Year'},
            color_discrete_map=self.president_colors
        )

        fig.update_traces(marker=dict(size=8, opacity=0.75))
        fig.update_layout(width=1000, height=700)

        fig.write_html(self.output_dir / filename)
        logger.info(f"Saved {filename}")

    def plot_by_president_panel(self, coordinates: np.ndarray,
                                metadata: pd.DataFrame,
                                filename: str = 'president_panels.png'):
        """Create per-president panels colored by year"""

        n_points = min(len(metadata), len(coordinates))
        if n_points == 0:
            logger.warning("No points available for president panel plot.")
            return

        plot_df = metadata.iloc[:n_points].copy().reset_index(drop=True)
        coords = coordinates[:n_points]
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        plot_df['year_num'] = pd.to_numeric(plot_df['year'], errors='coerce')

        presidents = [p for p in ['Clinton', 'Bush', 'Obama', 'Trump', 'Biden'] if p in plot_df['president'].dropna().unique()]
        if not presidents:
            logger.warning("No known presidents found for panel plot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        first_scatter = None
        for idx, president in enumerate(presidents):
            ax = axes[idx]
            mask = plot_df['president'] == president
            if mask.sum() == 0:
                continue

            ax.scatter(plot_df['x'], plot_df['y'], c='lightgray', alpha=0.25, s=18)

            color_vals = plot_df.loc[mask, 'year_num']
            if color_vals.notna().any():
                sc = ax.scatter(
                    plot_df.loc[mask, 'x'],
                    plot_df.loc[mask, 'y'],
                    c=color_vals,
                    cmap='viridis',
                    s=45,
                    alpha=0.85,
                    edgecolors='none'
                )
                if first_scatter is None:
                    first_scatter = sc
            else:
                ax.scatter(
                    plot_df.loc[mask, 'x'],
                    plot_df.loc[mask, 'y'],
                    c=self.president_colors.get(president, '#333333'),
                    s=45,
                    alpha=0.85,
                    edgecolors='none'
                )

            ax.set_title(f"{president} ({int(mask.sum())} orders)")
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')

        for j in range(len(presidents), len(axes)):
            axes[j].set_visible(False)

        if first_scatter is not None:
            fig.colorbar(first_scatter, ax=axes.tolist(), label='Year', shrink=0.85)

        plt.suptitle('Executive Orders by President (Colored by Year)')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi)
        plt.close()

        logger.info(f"Saved {filename}")

    def plot_centroid_trajectory(self, coordinates: np.ndarray,
                                 metadata: pd.DataFrame,
                                 filename: str = 'centroid_trajectory.png'):
        """Plot yearly centroid movement to show semantic drift trajectory"""

        n_points = min(len(metadata), len(coordinates))
        if n_points == 0:
            logger.warning("No points available for centroid trajectory plot.")
            return

        plot_df = metadata.iloc[:n_points].copy().reset_index(drop=True)
        coords = coordinates[:n_points]
        plot_df['x'] = coords[:, 0]
        plot_df['y'] = coords[:, 1]
        plot_df['year_num'] = pd.to_numeric(plot_df['year'], errors='coerce')
        plot_df = plot_df.dropna(subset=['year_num']).copy()
        if plot_df.empty:
            logger.warning("No valid year values found for centroid trajectory plot.")
            return

        plot_df['year_num'] = plot_df['year_num'].astype(int)
        years = sorted(plot_df['year_num'].unique())

        centroids = []
        centroid_years = []
        for year in years:
            year_df = plot_df[plot_df['year_num'] == year]
            if year_df.empty:
                continue
            centroids.append([year_df['x'].mean(), year_df['y'].mean()])
            centroid_years.append(year)

        if not centroids:
            logger.warning("No centroids computed for trajectory plot.")
            return

        centroids = np.asarray(centroids)

        plt.figure(figsize=(12, 8))
        plt.scatter(plot_df['x'], plot_df['y'], c='lightgray', alpha=0.25, s=18, label='All Orders')
        plt.plot(centroids[:, 0], centroids[:, 1], '-', color='darkred', linewidth=2, alpha=0.8, label='Centroid Path')

        sc = plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c=centroid_years,
            cmap='viridis',
            s=95,
            edgecolors='black',
            linewidth=0.7,
            label='Yearly Centroid'
        )

        for i, year in enumerate(centroid_years):
            if i % 2 == 0:
                plt.annotate(str(year), (centroids[i, 0], centroids[i, 1]), fontsize=8, alpha=0.85)

        plt.colorbar(sc, label='Year')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Semantic Evolution of Executive Orders (Yearly Centroid Trajectory)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.dpi)
        plt.close()

        logger.info(f"Saved {filename}")