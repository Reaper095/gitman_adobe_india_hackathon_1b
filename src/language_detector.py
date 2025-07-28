"""
Language detection module for Advanced Persona-Driven Document Intelligence v3.0
Handles multilingual text analysis and language detection.
"""

import re
import sys
from typing import Dict, Set

# Multilingual support imports
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available, using basic language detection", file=sys.stderr)

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False
    print("Warning: langid not available, using basic language detection", file=sys.stderr)

from .config import LANGUAGE_PATTERNS


class LanguageDetector:
    """Handles language detection for multilingual text analysis."""
    
    def __init__(self):
        """Initialize the language detector."""
        self.language_patterns = LANGUAGE_PATTERNS
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        if not text or len(text.strip()) < 10:
            return 'en'  # Default to English for short texts
        
        # Try langdetect first
        if LANGDETECT_AVAILABLE:
            try:
                return detect(text)
            except:
                pass
        
        # Try langid as fallback
        if LANGID_AVAILABLE:
            try:
                lang, _ = langid.classify(text)
                return lang
            except:
                pass
        
        # Pattern-based detection as last resort
        return self._pattern_based_detection(text)
    
    def _pattern_based_detection(self, text: str) -> str:
        """Pattern-based language detection as fallback."""
        text_lower = text.lower()
        lang_scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                lang_scores[lang] = matches / len(text_lower)
        
        if lang_scores:
            return max(lang_scores, key=lang_scores.get)
        
        return 'en'  # Default to English
    
    def get_supported_languages(self) -> Set[str]:
        """Get the set of supported languages."""
        return set(self.language_patterns.keys())
    
    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self.language_patterns 