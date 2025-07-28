"""
Document extraction module for Advanced Persona-Driven Document Intelligence v3.0
Handles PDF processing and content extraction with multilingual support.
"""

import fitz  # PyMuPDF
from datetime import datetime
from typing import Dict, List
import sys

from .language_detector import LanguageDetector
from .config import HEADING_PATTERNS, HEADING_KEYWORDS


class DocumentExtractor:
    """Handles PDF document extraction and content analysis."""
    
    def __init__(self):
        """Initialize the document extractor."""
        self.language_detector = LanguageDetector()
    
    def extract_document_content_multilingual(self, pdf_path: str) -> Dict:
        """Extract document content with multilingual support."""
        try:
            doc = fitz.open(pdf_path)
            content = {
                'pages': [],
                'sections': [],
                'metadata': {
                    'total_pages': len(doc),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'detected_languages': set()
                }
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with formatting
                blocks = page.get_text("dict")
                page_content = {
                    'page_number': page_num + 1,
                    'text_blocks': [],
                    'headings': [],
                    'body_text': [],
                    'languages': set()
                }
                
                for block in blocks["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            line_spans = []
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    line_spans.append(span)
                            
                            if line_text.strip() and line_spans:
                                # Detect language for this text block
                                detected_lang = self.language_detector.detect_language(line_text.strip())
                                page_content['languages'].add(detected_lang)
                                content['metadata']['detected_languages'].add(detected_lang)
                                
                                # Analyze text formatting
                                primary_span = line_spans[0]
                                font_size = primary_span["size"]
                                is_bold = bool(primary_span["flags"] & 2**4)
                                
                                text_block = {
                                    'text': line_text.strip(),
                                    'font_size': round(font_size, 1),
                                    'is_bold': is_bold,
                                    'font_name': primary_span["font"],
                                    'bbox': primary_span["bbox"],
                                    'language': detected_lang
                                }
                                
                                page_content['text_blocks'].append(text_block)
                                
                                # Classify as heading or body text
                                if self._is_heading_advanced(text_block):
                                    page_content['headings'].append(text_block)
                                else:
                                    page_content['body_text'].append(text_block)
                
                content['pages'].append(page_content)
            
            doc.close()
            return content
            
        except Exception as e:
            print(f"Error extracting content from {pdf_path}: {str(e)}", file=sys.stderr)
            return {'pages': [], 'sections': [], 'metadata': {}}
    
    def _is_heading_advanced(self, text_block: Dict) -> bool:
        """Advanced heading detection with multiple criteria."""
        text = text_block['text']
        font_size = text_block['font_size']
        is_bold = text_block['is_bold']
        
        # Basic criteria
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Font size and formatting
        if font_size > 12 and is_bold:
            return True
        
        # Pattern matching
        import re
        for pattern in HEADING_PATTERNS:
            if re.match(pattern, text):
                return True
        
        # Keyword-based detection
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in HEADING_KEYWORDS):
            return True
        
        return False
    
    def _find_associated_content(self, heading: Dict, body_text: List[Dict], 
                               all_pages: List[Dict], current_page: int) -> str:
        """Find content associated with a heading."""
        associated_content = []
        
        # Look for content after the heading on the same page
        heading_bbox = heading['bbox']
        
        for text_block in body_text:
            text_bbox = text_block['bbox']
            
            # Check if text block is below the heading
            if text_bbox[1] > heading_bbox[1]:  # y-coordinate comparison
                associated_content.append(text_block['text'])
        
        # If no content found on same page, look on next page
        if not associated_content and current_page < len(all_pages):
            next_page = all_pages[current_page]  # current_page is 1-indexed
            for text_block in next_page.get('body_text', []):
                associated_content.append(text_block['text'])
        
        return ' '.join(associated_content[:3])  # Limit to first 3 blocks 