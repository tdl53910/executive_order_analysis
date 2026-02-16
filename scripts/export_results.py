#!/usr/bin/env python3
"""
Script to export analysis results to various formats (CSV, JSON, Excel)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsExporter:
    """Export analysis results to different formats"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embeddings_dir = Path(self.config['data']['embeddings_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
        
    def export_all(self):
        """Export all results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export metadata and statistics
        self._export_metadata(timestamp)
        
        # Export coordinates
        self._export_coordinates(timestamp)
        
        # Export cluster analysis
        self._export_clusters(timestamp)
        
        # Export semantic drift
        self._export_drift(timestamp)
        
        # Create summary report
        self._create_summary_report(timestamp)
        
        logger.info(f"All results exported to {self.export_dir}/")
    
    def _export_metadata(self, timestamp: str):
        """Export metadata with statistics"""
        
        metadata_path = self.embeddings_dir / 'embedding_metadata.csv'
        if not metadata_path.exists():
            logger.warning("Metadata not found")
            return
        
        df = pd.read_csv(metadata_path)
        
        # Add statistics
        stats_path = self.embeddings_dir.parent / 'processed' / 'preprocessing_stats.csv'
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
            # Merge on filename if possible
            if 'filename' in stats_df.columns:
                df = df.merge(stats_df, on='filename', how='left')
        
        # Export as CSV
        csv_path = self.export_dir / f'metadata_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        # Export as Excel
        excel_path = self.export_dir / f'metadata_{timestamp}.xlsx'
        df.to_excel(excel_path, index=False)
        
        # Export as JSON
        json_path = self.export_dir / f'metadata_{timestamp}.json'
        df.to_json(json_path, orient='records', indent=2)
        
        logger.info(f"Exported metadata ({len(df)} records)")
    
    def _export_coordinates(self, timestamp: str):
        """Export reduced coordinates"""
        
        for method in ['pca', 'tsne', 'umap']:
            coords_path = self.output_dir / f'{method}_coordinates.npy'
            if not coords_path.exists():
                continue
            
            coords = np.load(coords_path)
            
            # Load metadata to add labels
            metadata_path = self.embeddings_dir / 'embedding_metadata.csv'
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path)
                
                # Create DataFrame with coordinates
                df = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'filename': metadata['filename'],
                    'president': metadata['president'],
                    'year': metadata['year']
                })
                
                # Export
                csv_path = self.export_dir / f'{method}_coordinates_{timestamp}.csv'
                df.to_csv(csv_path, index=False)
                
                logger.info(f"Exported {method} coordinates ({len(df)} points)")
    
    def _export_clusters(self, timestamp: str):
        """Export cluster assignments"""
        
        for method in ['kmeans', 'dbscan']:
            labels_path = self.output_dir / f'{method}_labels.npy'
            if not labels_path.exists():
                continue
            
            labels = np.load(labels_path)
            
            # Load metadata
            metadata_path = self.embeddings_dir / 'embedding_metadata.csv'
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path)
                
                df = pd.DataFrame({
                    'filename': metadata['filename'],
                    'president': metadata['president'],
                    'year': metadata['year'],
                    'cluster': labels
                })
                
                # Add cluster statistics
                cluster_stats = df.groupby('cluster').agg({
                    'filename': 'count',
                    'president': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
                }).rename(columns={'filename': 'count'})
                
                # Export
                csv_path = self.export_dir / f'{method}_clusters_{timestamp}.csv'
                df.to_csv(csv_path, index=False)
                
                stats_path = self.export_dir / f'{method}_cluster_stats_{timestamp}.csv'
                cluster_stats.to_csv(stats_path)
                
                logger.info(f"Exported {method} clusters")
    
    def _export_drift(self, timestamp: str):
        """Export semantic drift analysis"""
        
        drift_path = self.output_dir / 'semantic_drift.csv'
        if not drift_path.exists():
            return
        
        drift_df = pd.read_csv(drift_path)
        
        # Export in multiple formats
        csv_path = self.export_dir / f'semantic_drift_{timestamp}.csv'
        drift_df.to_csv(csv_path, index=False)
        
        excel_path = self.export_dir / f'semantic_drift_{timestamp}.xlsx'
        drift_df.to_excel(excel_path, index=False)
        
        # Create JSON with additional metadata
        drift_data = {
            'transitions': drift_df.to_dict('records'),
            'summary': {
                'mean_drift': drift_df['drift'].mean(),
                'max_drift': drift_df['drift'].max(),
                'min_drift': drift_df['drift'].min(),
                'total_change': drift_df['drift'].sum()
            }
        }
        
        json_path = self.export_dir / f'semantic_drift_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(drift_data, f, indent=2)
        
        logger.info("Exported semantic drift analysis")
    
    def _create_summary_report(self, timestamp: str):
        """Create a human-readable summary report"""
        
        report = []
        report.append("# Executive Order Analysis Summary Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Load metadata
        metadata_path = self.embeddings_dir / 'embedding_metadata.csv'
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            
            report.append("## Dataset Overview")
            report.append(f"- Total executive orders analyzed: {len(metadata)}")
            report.append(f"- Date range: {metadata['year'].min()} - {metadata['year'].max()}")
            report.append("")
            
            # By president
            report.append("### Distribution by President")
            for president, count in metadata['president'].value_counts().items():
                report.append(f"- {president}: {count} orders ({count/len(metadata)*100:.1f}%)")
            report.append("")
        
        # Load drift analysis
        drift_path = self.output_dir / 'semantic_drift.csv'
        if drift_path.exists():
            drift_df = pd.read_csv(drift_path)
            
            report.append("## Semantic Drift Analysis")
            report.append(f"- Mean semantic drift between years: {drift_df['drift'].mean():.3f}")
            report.append(f"- Maximum drift: {drift_df['drift'].max():.3f}")
            
            # Find most significant transitions
            top_drifts = drift_df.nlargest(3, 'drift')
            report.append("\n### Most Significant Transitions:")
            for _, row in top_drifts.iterrows():
                report.append(f"- {row['from_year']} → {row['to_year']}: {row['drift']:.3f}")
            report.append("")
        
        # Load cluster stats
        for method in ['kmeans']:
            stats_path = self.export_dir / f'{method}_cluster_stats_{timestamp}.csv'
            if stats_path.exists():
                cluster_stats = pd.read_csv(stats_path)
                
                report.append(f"## {method.title()} Clustering Results")
                report.append(f"- Number of clusters: {len(cluster_stats)}")
                report.append("\n### Cluster Sizes:")
                for _, row in cluster_stats.iterrows():
                    report.append(f"- Cluster {row['cluster']}: {row['count']} orders (primarily {row['president']})")
                report.append("")
        
        # Save report
        report_path = self.export_dir / f'summary_report_{timestamp}.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Export analysis results')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    exporter = ResultsExporter(args.config)
    exporter.export_all()

if __name__ == "__main__":
    main()