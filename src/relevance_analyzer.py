"""
Relevance analysis module for Advanced Persona-Driven Document Intelligence v3.0
Handles semantic relevance calculation and content scoring with multilingual support.
"""

import sys
from typing import Dict, List
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from .language_detector import LanguageDetector
from .config import PERSONA_KNOWLEDGE, JOB_PATTERNS, SECTION_KEYWORDS, TECHNICAL_TERMS


class RelevanceAnalyzer:
    """Handles semantic relevance calculation and content analysis."""
    
    def __init__(self, sentence_model=None):
        """Initialize the relevance analyzer."""
        self.language_detector = LanguageDetector()
        self.sentence_model = sentence_model
    
    def calculate_semantic_relevance_multilingual(self, text: str, persona: str, job: str) -> float:
        """Calculate semantic relevance with multilingual support."""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Detect language
        detected_lang = self.language_detector.detect_language(text)
        text_lower = text.lower()
        relevance_score = 0.0
        
        # Get language-specific keywords
        keywords = self.get_keywords_for_language(persona, job, detected_lang)
        
        # 1. Keyword-based scoring (40% weight)
        keyword_score = 0
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                keyword_score += weight
        
        # Normalize keyword score
        max_keyword_score = sum(keywords.values()) if keywords else 1
        keyword_score = (keyword_score / max_keyword_score) * 40
        relevance_score += keyword_score
        
        # 2. Multilingual sentence similarity (30% weight)
        if self.sentence_model:
            try:
                # Create reference text in detected language
                reference_text = f"{persona} {job} {' '.join(keywords.keys())}"
                
                # Calculate similarity
                embeddings = self.sentence_model.encode([text, reference_text])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                relevance_score += similarity * 30
            except Exception as e:
                print(f"Warning: Multilingual similarity calculation failed: {e}", file=sys.stderr)
        
        # 3. Language-specific content analysis (20% weight)
        structure_score = self._calculate_structure_score(text_lower, detected_lang)
        relevance_score += structure_score
        
        # 4. Text quality analysis (10% weight)
        quality_score = self._calculate_quality_score(text, detected_lang, persona)
        relevance_score += quality_score
        
        return min(100, max(0, relevance_score))
    
    def get_keywords_for_language(self, persona: str, job: str, language: str) -> Dict[str, int]:
        """Get keywords for the specified persona, job, and language."""
        persona_keywords = PERSONA_KNOWLEDGE.get(persona, {}).get('keywords', {})
        job_keywords = JOB_PATTERNS.get(job, {}).get('keywords', {})
        
        # Get language-specific keywords
        persona_lang_keywords = persona_keywords.get(language, persona_keywords.get('en', {}))
        job_lang_keywords = job_keywords.get(language, job_keywords.get('en', {}))
        
        # Combine keywords
        combined_keywords = {}
        combined_keywords.update(persona_lang_keywords)
        combined_keywords.update(job_lang_keywords)
        
        return combined_keywords
    
    def _calculate_structure_score(self, text_lower: str, language: str) -> float:
        """Calculate structure score based on section keywords."""
        structure_score = 0
        
        # Check for relevant sections in multiple languages
        lang_sections = SECTION_KEYWORDS.get(language, SECTION_KEYWORDS['en'])
        for section in lang_sections:
            if section in text_lower:
                structure_score += 5
        
        return min(structure_score, 20)
    
    def _calculate_quality_score(self, text: str, language: str, persona: str) -> float:
        """Calculate quality score based on text characteristics."""
        quality_score = 0
        
        # Length appropriateness
        if 50 <= len(text) <= 500:
            quality_score += 5
        elif 20 <= len(text) <= 1000:
            quality_score += 3
        
        # Sentence structure
        sentences = sent_tokenize(text)
        if 1 <= len(sentences) <= 5:
            quality_score += 3
        
        # Language-specific technical terms
        if persona in ['researcher', 'developer', 'analyst']:
            lang_terms = TECHNICAL_TERMS.get(language, TECHNICAL_TERMS['en'])
            if any(term in text.lower() for term in lang_terms):
                quality_score += 2
        
        return quality_score
    
    def refine_content_for_persona(self, content: str, persona: str, job: str) -> str:
        """Refine content specifically for the target persona and job."""
        # Extract key sentences
        sentences = sent_tokenize(content)
        
        # Score sentences based on persona and job relevance
        scored_sentences = []
        for sentence in sentences:
            score = self.calculate_semantic_relevance_multilingual(sentence, persona, job)
            scored_sentences.append((sentence, score))
        
        # Sort by relevance and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences (limit to 3-5 sentences)
        max_sentences = 4 if persona in ['researcher', 'analyst'] else 3
        selected_sentences = [s[0] for s in scored_sentences[:max_sentences] if s[1] > 20]
        
        return ' '.join(selected_sentences) if selected_sentences else content[:300]
    
    def generate_selection_reasoning(self, section: Dict, persona: str, job: str) -> str:
        """Generate reasoning for why a section was selected."""
        title = section.get('section_title', '')
        relevance_score = section.get('relevance_score', 0)
        content = section.get('content', '')
        language = section.get('language', 'en')
        
        # Get language-specific keywords
        keywords = self.get_keywords_for_language(persona, job, language)
        
        # Analyze why this section is relevant
        reasons = []
        
        # Check for keyword matches
        title_lower = title.lower()
        content_lower = content.lower()
        
        matched_keywords = []
        for keyword, weight in keywords.items():
            if keyword in title_lower or keyword in content_lower:
                matched_keywords.append(keyword)
        
        if matched_keywords:
            reasons.append(f"Contains relevant keywords: {', '.join(matched_keywords[:3])}")
        
        # Language-specific reasoning
        lang_reasons = {
            'en': f"Content in English with relevance score {relevance_score}",
            'es': f"Contenido en español con puntuación de relevancia {relevance_score}",
            'fr': f"Contenu en français avec score de pertinence {relevance_score}",
            'de': f"Inhalt auf Deutsch mit Relevanzbewertung {relevance_score}",
            'hi': f"हिंदी में सामग्री जिसकी प्रासंगिकता स्कोर {relevance_score} है"
        }
        
        reasons.append(lang_reasons.get(language, lang_reasons['en']))
        
        # Check content characteristics
        if len(content) > 100:
            reasons.append("Contains substantial content for detailed analysis")
        
        if relevance_score > 50:
            reasons.append(f"High relevance score ({relevance_score}) indicates strong semantic alignment")
        elif relevance_score > 30:
            reasons.append(f"Moderate relevance score ({relevance_score}) shows good content value")
        
        return "; ".join(reasons) 