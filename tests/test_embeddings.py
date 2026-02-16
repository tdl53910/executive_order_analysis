"""
Unit tests for the embeddings module
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.embeddings.embedding_generator import EmbeddingGenerator

class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for EmbeddingGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.generator = EmbeddingGenerator()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_model_loading(self):
        """Test that model loads successfully"""
        self.assertIsNotNone(self.generator.model)
    
    def test_embedding_generation(self):
        """Test generating embeddings for sample text"""
        test_texts = [
            "This is a test executive order about national security.",
            "This order concerns economic policy and regulation."
        ]
        
        embeddings = self.generator.model.encode(test_texts)
        
        # Check shape
        self.assertEqual(len(embeddings), 2)
        self.assertGreater(embeddings.shape[1], 0)
        
        # Check that different texts have different embeddings
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        self.assertLess(similarity, 1.0)  # Not identical
    
    def test_batch_processing(self):
        """Test batch processing of texts"""
        texts = [f"Test text {i}" for i in range(10)]
        
        # Process in batches
        all_embeddings = []
        batch_size = 3
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.generator.model.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        self.assertEqual(len(embeddings), 10)

if __name__ == '__main__':
    unittest.main()