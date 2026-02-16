"""
Unit tests for the scraper module
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

from src.scraper.federal_register_scraper import FederalRegisterScraper

class TestFederalRegisterScraper(unittest.TestCase):
    """Test cases for FederalRegisterScraper"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.scraper = FederalRegisterScraper()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    @patch('requests.Session.get')
    def test_fetch_api_response(self, mock_get):
        """Test API response fetching"""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {'results': []}
        mock_get.return_value = mock_response
        
        result = self.scraper._fetch_api_response("http://test.com")
        self.assertEqual(result, {'results': []})
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        test_cases = [
            ("Test File Name", "Test_File_Name"),
            ("Test/File:Name*", "Test_File_Name_"),
            ("  Spaces  ", "Spaces"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.scraper._sanitize_filename(input_name)
                self.assertEqual(result, expected)
    
    @patch('src.scraper.federal_register_scraper.FederalRegisterScraper._fetch_html_text')
    def test_process_order(self, mock_fetch):
        """Test processing a single order"""
        mock_fetch.return_value = "Test content"
        
        test_doc = {
            'executive_order_number': '12345',
            'title': 'Test Order',
            'publication_date': '2024-01-01',
            'president': 'Test President',
            'html_url': 'http://test.com',
            'pdf_url': ''
        }
        
        result = self.scraper._process_order(test_doc)
        self.assertIsNotNone(result)
        self.assertEqual(result['eo_number'], '12345')
        self.assertEqual(result['president'], 'Test President')

if __name__ == '__main__':
    unittest.main()