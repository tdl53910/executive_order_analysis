"""
Unit tests for the preprocessing module
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from src.preprocessing.text_cleaner import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.preprocessor = TextPreprocessor()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        test_text = "EXECUTIVE ORDER 13769 — Protecting the Nation From Foreign Terrorist Entry"
        
        processed, stats = self.preprocessor.preprocess_text(test_text)
        
        # Check that EO number was removed
        self.assertNotIn("13769", processed)
        
        # Check lowercase conversion
        self.assertTrue(processed.islower())
        
        # Check stats
        self.assertIn('original_words', stats)
        self.assertIn('processed_words', stats)
        self.assertGreater(stats['original_words'], 0)
    
    def test_preprocess_text_with_stopwords(self):
        """Test stopword removal"""
        test_text = "The president shall hereby order that the policy is effective"
        
        processed, _ = self.preprocessor.preprocess_text(test_text)
        
        # Common stopwords should be removed
        common_stopwords = ['the', 'shall', 'hereby', 'that', 'is']
        for word in common_stopwords:
            self.assertNotIn(word, processed)
    
    def test_preprocess_text_empty(self):
        """Test preprocessing empty text"""
        processed, stats = self.preprocessor.preprocess_text("")
        
        self.assertEqual(processed, "")
        self.assertEqual(stats['original_words'], 0)
    
    def test_calculate_stats(self):
        """Test statistics calculation"""
        test_text = "This is a test sentence. This is another sentence."
        
        processed, stats = self.preprocessor.preprocess_text(test_text)
        
        self.assertGreater(stats['original_words'], 0)
        self.assertGreater(stats['sentence_count'], 1)
        self.assertGreater(stats['flesch_reading_ease'], 0)

if __name__ == '__main__':
    unittest.main()