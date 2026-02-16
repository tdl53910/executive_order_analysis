# src/scraper/federal_register_scraper.py

import requests
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from tqdm import tqdm
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederalRegisterScraper:
    """Scraper for executive orders from the Federal Register API"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.base_url = "https://www.federalregister.gov/api/v1/documents.json"
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ExecutiveOrderResearch/1.0)',
            'Accept': 'application/json'
        })
        
    def scrape_orders(self, start_year: int = None, end_year: int = None):
        """Scrape all executive orders within year range"""
        
        if start_year is None:
            start_year = self.config['scraping']['start_year']
        if end_year is None:
            end_year = self.config['scraping']['end_year']
            
        logger.info(f"Scraping executive orders from {start_year} to {end_year}")
        
        page = 1
        total_downloaded = 0
        metadata = []
        
        while True:
            # Build API request
            params = {
                'per_page': self.config['scraping']['batch_size'],
                'page': page,
                'conditions[presidential_document_type]': 'executive_order',
                'conditions[publication_date][gte]': f"{start_year}-01-01",
                'conditions[publication_date][lte]': f"{end_year}-12-31",
                'order': 'newest'
            }
            
            # Make request with retry logic
            for attempt in range(self.config['scraping']['max_retries']):
                try:
                    response = self.session.get(self.base_url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as e:
                    if attempt == self.config['scraping']['max_retries'] - 1:
                        logger.error(f"Failed after {attempt+1} attempts: {e}")
                        return
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            results = data.get('results', [])
            if not results:
                logger.info(f"No more results at page {page}")
                break
                
            logger.info(f"Processing page {page} ({len(results)} documents)")
            
            # Process each order
            for doc in tqdm(results, desc=f"Page {page}"):
                order_data = self._process_order(doc)
                if order_data:
                    metadata.append(order_data)
                    total_downloaded += 1
            
            # Rate limiting
            time.sleep(self.config['scraping']['delay'])
            page += 1
        
        # Save metadata
        self._save_metadata(metadata)
        logger.info(f"Scraping complete! Downloaded {total_downloaded} orders")
        
    def _process_order(self, doc: Dict) -> Optional[Dict]:
        """Process a single executive order document"""
        
        eo_number = doc.get('executive_order_number', 'NA')
        title = doc.get('title', 'Untitled')
        date = doc.get('publication_date', '')
        president = doc.get('president', 'Unknown')
        html_url = doc.get('html_url', '')
        pdf_url = doc.get('pdf_url', '')
        
        # Create safe filename
        safe_title = self._sanitize_filename(title)[:50]
        filename = f"EO_{eo_number}_{date}_{safe_title}.txt"
        filepath = self.raw_dir / filename
        
        # Try to get text from HTML
        text_content = None
        if html_url:
            text_content = self._fetch_html_text(html_url)
        
        if text_content is None and pdf_url:
            # Note: For PDFs, you'd need to implement PDF extraction
            # This could use PyPDF2 or pdfplumber
            logger.warning(f"PDF extraction not implemented for {eo_number}")
            text_content = "PDF_CONTENT_NOT_EXTRACTED"
        
        if text_content is None:
            text_content = "CONTENT_NOT_AVAILABLE"
        
        # Save text file
        filepath.write_text(text_content, encoding='utf-8')
        
        return {
            'eo_number': eo_number,
            'president': president,
            'date': date,
            'title': title,
            'filename': filename,
            'html_url': html_url,
            'pdf_url': pdf_url
        }
    
    def _fetch_html_text(self, url: str) -> Optional[str]:
        """Fetch and extract text from HTML page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove navigation, headers, footers
            for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style']):
                element.decompose()
            
            # Get main content - adjust selectors based on Federal Register structure
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', {'class': 'content'}) or 
                soup.body
            )
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
                # Clean up whitespace
                text = ' '.join(text.split())
                return text
            
        except Exception as e:
            logger.warning(f"Failed to fetch HTML from {url}: {e}")
        
        return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Convert string to safe filename"""
        import re
        # Replace invalid characters with underscore
        filename = re.sub(r'[^\w\s-]', '_', filename)
        # Replace spaces with underscore
        filename = re.sub(r'[-\s]+', '_', filename)
        return filename.strip('_')
    
    def _save_metadata(self, metadata: List[Dict]):
        """Save metadata to CSV file"""
        if not metadata:
            return
            
        csv_path = self.raw_dir / 'metadata.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
        
        logger.info(f"Metadata saved to {csv_path}")