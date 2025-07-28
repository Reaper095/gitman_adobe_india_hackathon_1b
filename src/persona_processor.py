"""
Main persona processor module for Advanced Persona-Driven Document Intelligence v3.0
Orchestrates the entire document analysis process with multilingual support.
"""

import os
import time
import sys
from typing import Dict, List, Tuple
from datetime import datetime

from .document_extractor import DocumentExtractor
from .relevance_analyzer import RelevanceAnalyzer
from .language_detector import LanguageDetector
from .config import PERSONA_KNOWLEDGE, JOB_PATTERNS


class AdvancedPersonaIntelligence:
    """Main processor for advanced persona-driven document intelligence."""
    
    def __init__(self):
        """Initialize the advanced persona intelligence system."""
        self.start_time = time.time()
        
        # Initialize components
        self.document_extractor = DocumentExtractor()
        self.language_detector = LanguageDetector()
        
        # Initialize sentence transformer for semantic similarity
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Loaded multilingual sentence transformer", file=sys.stderr)
        except Exception as e:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded English sentence transformer as fallback", file=sys.stderr)
            except Exception as e:
                self.sentence_model = None
                print(f"Warning: Sentence transformer not available: {e}", file=sys.stderr)
        
        # Initialize relevance analyzer with sentence model
        self.relevance_analyzer = RelevanceAnalyzer(self.sentence_model)
    
    def check_time_limit(self, stage: str) -> bool:
        """Check if processing time limit is exceeded."""
        elapsed = time.time() - self.start_time
        if elapsed > 300:  # 5 minutes limit
            print(f"Time limit exceeded at {stage}", file=sys.stderr)
            return False
        return True
    
    def extract_relevant_sections_multilingual(self, content: Dict, persona: str, job: str) -> List[Dict]:
        """Extract relevant sections with multilingual support."""
        relevant_sections = []
        
        for page_content in content['pages']:
            page_num = page_content['page_number']
            
            # Process headings and their associated content
            for heading in page_content['headings']:
                if not self.check_time_limit(f"section analysis page {page_num}"):
                    break
                
                # Find content associated with this heading
                associated_content = self.document_extractor._find_associated_content(
                    heading, page_content['body_text'], content['pages'], page_num
                )
                
                # Calculate relevance with multilingual support
                combined_text = f"{heading['text']} {associated_content}"
                relevance_score = self.relevance_analyzer.calculate_semantic_relevance_multilingual(
                    combined_text, persona, job
                )
                
                if relevance_score > 15:  # Lower threshold for better coverage
                    section = {
                        'document': content.get('metadata', {}).get('filename', 'Unknown'),
                        'page': page_num,
                        'section_title': heading['text'],
                        'content': associated_content,
                        'relevance_score': float(round(relevance_score, 2)),
                        'font_size': float(heading['font_size']),
                        'is_bold': heading['is_bold'],
                        'language': heading.get('language', 'en')
                    }
                    relevant_sections.append(section)
            
            # Also check body text for relevant content
            for text_block in page_content['body_text']:
                if not self.check_time_limit(f"body text analysis page {page_num}"):
                    break
                
                relevance_score = self.relevance_analyzer.calculate_semantic_relevance_multilingual(
                    text_block['text'], persona, job
                )
                
                if relevance_score > 20:  # Lower threshold for body text
                    section = {
                        'document': content.get('metadata', {}).get('filename', 'Unknown'),
                        'page': page_num,
                        'section_title': f"Content from page {page_num}",
                        'content': text_block['text'],
                        'relevance_score': float(round(relevance_score, 2)),
                        'font_size': float(text_block['font_size']),
                        'is_bold': text_block['is_bold'],
                        'language': text_block.get('language', 'en')
                    }
                    relevant_sections.append(section)
        
        return relevant_sections
    
    def create_subsection_analysis_advanced(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Advanced subsection analysis with persona-specific insights."""
        subsections = []
        
        for section in sections:
            if not self.check_time_limit("subsection analysis"):
                break
            
            content = section.get('content', '')
            if not content or len(content.strip()) < 20:
                continue
            
            # Advanced content refinement
            refined_text = self.relevance_analyzer.refine_content_for_persona(content, persona, job)
            
            if refined_text:
                subsection = {
                    'document': section['document'],
                    'page': section['page'],
                    'section_title': section['section_title'],
                    'refined_text': refined_text,
                    'relevance_score': section['relevance_score'],
                    'persona_focus': self._get_persona_focus_areas(persona),
                    'job_alignment': self._get_job_alignment(job)
                }
                subsections.append(subsection)
        
        return subsections
    
    def _get_persona_focus_areas(self, persona: str) -> List[str]:
        """Get focus areas for the persona."""
        return PERSONA_KNOWLEDGE.get(persona, {}).get('focus_areas', [])
    
    def _get_job_alignment(self, job: str) -> str:
        """Get job alignment description."""
        return JOB_PATTERNS.get(job, {}).get('focus', 'General content')
    
    def rank_and_filter_results_advanced(self, sections: List[Dict], subsections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Advanced ranking and filtering with improved algorithms."""
        # Rank sections by relevance score
        ranked_sections = sorted(sections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply additional filtering
        filtered_sections = []
        for section in ranked_sections:
            # Filter out very short or very long sections
            content_length = len(section.get('content', ''))
            if 50 <= content_length <= 2000:
                filtered_sections.append(section)
        
        # Rank subsections
        ranked_subsections = sorted(subsections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply quality filtering to subsections
        filtered_subsections = []
        for subsection in ranked_subsections:
            refined_length = len(subsection.get('refined_text', ''))
            if 30 <= refined_length <= 1000:
                filtered_subsections.append(subsection)
        
        # Limit results for better quality
        max_sections = 10
        max_subsections = 15
        
        return filtered_sections[:max_sections], filtered_subsections[:max_subsections]
    
    def process_documents_multilingual(self, docs_path: str, persona: str, job: str) -> Dict:
        """Process documents with multilingual support."""
        try:
            if not os.path.exists(docs_path):
                raise FileNotFoundError(f"Input directory not found: {docs_path}")
            
            pdf_files = [f for f in os.listdir(docs_path) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                return self._create_empty_result(persona, job, "No PDF files found")
            
            print(f"Processing documents for persona: {persona}, job: {job}")
            
            all_sections = []
            all_subsections = []
            detected_languages = set()
            
            for pdf_file in pdf_files:
                if not self.check_time_limit(f"document {pdf_file}"):
                    break
                
                pdf_path = os.path.join(docs_path, pdf_file)
                print(f"Processing: {pdf_file}")
                
                # Extract content with multilingual support
                content = self.document_extractor.extract_document_content_multilingual(pdf_path)
                content['metadata']['filename'] = pdf_file
                
                # Track detected languages
                detected_languages.update(content['metadata'].get('detected_languages', set()))
                
                # Extract relevant sections with multilingual support
                sections = self.extract_relevant_sections_multilingual(content, persona, job)
                all_sections.extend(sections)
                
                # Create subsection analysis
                subsections = self.create_subsection_analysis_advanced(sections, persona, job)
                all_subsections.extend(subsections)
            
            # Rank and filter results
            ranked_sections, ranked_subsections = self.rank_and_filter_results_advanced(all_sections, all_subsections)
            
            # Prepare output format with multilingual information
            extracted_sections = []
            for i, section in enumerate(ranked_sections):
                # Generate multilingual reasoning
                selection_reasoning = self.relevance_analyzer.generate_selection_reasoning(section, persona, job)
                
                extracted_sections.append({
                    "document": section['document'],
                    "page": section['page'],
                    "section_title": section['section_title'],
                    "importance_rank": i + 1,
                    "relevance_score": float(section['relevance_score']),
                    "selection_reasoning": selection_reasoning,
                    "content_preview": section.get('content', '')[:200] + "..." if len(section.get('content', '')) > 200 else section.get('content', ''),
                    "language": section.get('language', 'en')
                })
            
            subsection_analysis = []
            for subsection in ranked_subsections:
                # Generate multilingual reasoning
                selection_reasoning = self.relevance_analyzer.generate_selection_reasoning(subsection, persona, job)
                
                subsection_analysis.append({
                    "document": subsection['document'],
                    "page": subsection['page'],
                    "section_title": subsection['section_title'],
                    "refined_text": subsection['refined_text'],
                    "relevance_score": float(subsection['relevance_score']),
                    "selection_reasoning": selection_reasoning,
                    "persona_focus": subsection['persona_focus'],
                    "job_alignment": subsection['job_alignment'],
                    "language": subsection.get('language', 'en')
                })
            
            total_time = time.time() - self.start_time
            print(f"Multilingual processing completed in {total_time:.2f} seconds")
            print(f"Detected languages: {', '.join(detected_languages)}")
            
            return {
                "metadata": {
                    "input_documents": pdf_files,
                    "persona": persona,
                    "job_to_be_done": job,
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_sections_found": len(extracted_sections),
                    "total_subsections_found": len(subsection_analysis),
                    "processing_time_seconds": round(total_time, 2),
                    "algorithm_version": "3.0_multilingual",
                    "detected_languages": list(detected_languages),
                    "multilingual_support": True
                },
                "extracted_sections": extracted_sections,
                "subsection_analysis": subsection_analysis
            }
            
        except Exception as e:
            print(f"Error processing documents: {str(e)}", file=sys.stderr)
            return self._create_empty_result(persona, job, str(e))
    
    def _create_empty_result(self, persona: str, job: str, error_msg: str = "") -> Dict:
        """Create empty result structure."""
        return {
            "metadata": {
                "input_documents": [],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_found": 0,
                "total_subsections_found": 0,
                "processing_time_seconds": 0,
                "error": error_msg,
                "algorithm_version": "3.0_multilingual"
            },
            "extracted_sections": [],
            "subsection_analysis": []
        } 