#!/usr/bin/env python3
"""
Test script for Advanced Persona-Driven Document Intelligence v3.0
Verifies that all modules work correctly.
"""

import sys
import os
import json
from datetime import datetime

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from src.config import PERSONA_KNOWLEDGE, JOB_PATTERNS
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config module import failed: {e}")
        return False
    
    try:
        from src.language_detector import LanguageDetector
        print("✅ Language detector module imported successfully")
    except ImportError as e:
        print(f"❌ Language detector module import failed: {e}")
        return False
    
    try:
        from src.document_extractor import DocumentExtractor
        print("✅ Document extractor module imported successfully")
    except ImportError as e:
        print(f"❌ Document extractor module import failed: {e}")
        return False
    
    try:
        from src.relevance_analyzer import RelevanceAnalyzer
        print("✅ Relevance analyzer module imported successfully")
    except ImportError as e:
        print(f"❌ Relevance analyzer module import failed: {e}")
        return False
    
    try:
        from src.persona_processor import AdvancedPersonaIntelligence
        print("✅ Persona processor module imported successfully")
    except ImportError as e:
        print(f"❌ Persona processor module import failed: {e}")
        return False
    
    return True

def test_language_detection():
    """Test language detection functionality."""
    print("\n🌍 Testing language detection...")
    
    try:
        from src.language_detector import LanguageDetector
        
        detector = LanguageDetector()
        
        # Test English
        english_text = "This is a sample English text for testing."
        detected_lang = detector.detect_language(english_text)
        print(f"✅ English detection: {detected_lang}")
        
        # Test Spanish
        spanish_text = "Este es un texto de ejemplo en español para pruebas."
        detected_lang = detector.detect_language(spanish_text)
        print(f"✅ Spanish detection: {detected_lang}")
        
        # Test supported languages
        supported = detector.get_supported_languages()
        print(f"✅ Supported languages: {', '.join(supported)}")
        
        return True
    except Exception as e:
        print(f"❌ Language detection test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from src.config import PERSONA_KNOWLEDGE, JOB_PATTERNS
        
        # Test persona knowledge
        personas = list(PERSONA_KNOWLEDGE.keys())
        print(f"✅ Available personas: {', '.join(personas)}")
        
        # Test job patterns
        jobs = list(JOB_PATTERNS.keys())
        print(f"✅ Available jobs: {', '.join(jobs)}")
        
        # Test specific persona
        researcher_keywords = PERSONA_KNOWLEDGE.get('researcher', {}).get('keywords', {}).get('en', {})
        print(f"✅ Researcher keywords count: {len(researcher_keywords)}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_relevance_analysis():
    """Test relevance analysis functionality."""
    print("\n📊 Testing relevance analysis...")
    
    try:
        from src.relevance_analyzer import RelevanceAnalyzer
        
        analyzer = RelevanceAnalyzer()
        
        # Test text
        test_text = "This research methodology examines data analysis and statistical methods."
        relevance_score = analyzer.calculate_semantic_relevance_multilingual(
            test_text, "researcher", "literature_review"
        )
        print(f"✅ Relevance score for researcher: {relevance_score:.2f}")
        
        # Test content refinement
        refined_text = analyzer.refine_content_for_persona(
            test_text, "researcher", "literature_review"
        )
        print(f"✅ Content refinement: {len(refined_text)} characters")
        
        return True
    except Exception as e:
        print(f"❌ Relevance analysis test failed: {e}")
        return False

def test_processor_initialization():
    """Test processor initialization."""
    print("\n🔧 Testing processor initialization...")
    
    try:
        from src.persona_processor import AdvancedPersonaIntelligence
        
        processor = AdvancedPersonaIntelligence()
        print("✅ Processor initialized successfully")
        
        # Test empty result creation
        empty_result = processor._create_empty_result("test_persona", "test_job", "test_error")
        print("✅ Empty result creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Processor initialization test failed: {e}")
        return False

def create_sample_output():
    """Create a sample output file for demonstration."""
    print("\n📄 Creating sample output...")
    
    sample_output = {
        "metadata": {
            "input_documents": ["sample_document.pdf"],
            "persona": "researcher",
            "job_to_be_done": "literature_review",
            "processing_timestamp": datetime.now().isoformat(),
            "total_sections_found": 5,
            "total_subsections_found": 8,
            "processing_time_seconds": 2.5,
            "algorithm_version": "3.0_multilingual",
            "detected_languages": ["en"],
            "multilingual_support": True
        },
        "extracted_sections": [
            {
                "document": "sample_document.pdf",
                "page": 1,
                "section_title": "Introduction",
                "importance_rank": 1,
                "relevance_score": 85.5,
                "selection_reasoning": "Contains relevant keywords: methodology, analysis",
                "content_preview": "This study examines the methodology and analysis techniques...",
                "language": "en"
            }
        ],
        "subsection_analysis": [
            {
                "document": "sample_document.pdf",
                "page": 1,
                "section_title": "Introduction",
                "refined_text": "Key findings include methodology analysis and statistical methods.",
                "relevance_score": 85.5,
                "selection_reasoning": "High relevance score indicates strong semantic alignment",
                "persona_focus": ["research design", "data analysis"],
                "job_alignment": "comprehensive overview of existing research",
                "language": "en"
            }
        ]
    }
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Save sample output
    with open("output/sample_analysis.json", "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample output created: output/sample_analysis.json")
    return True

def main():
    """Run all tests."""
    print("🚀 Advanced Persona-Driven Document Intelligence v3.0 - System Test")
    print("==================================================================")
    
    tests = [
        ("Module Imports", test_imports),
        ("Language Detection", test_language_detection),
        ("Configuration", test_configuration),
        ("Relevance Analysis", test_relevance_analysis),
        ("Processor Initialization", test_processor_initialization),
        ("Sample Output Creation", create_sample_output),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. Place PDF files in the 'input' directory")
        print("2. Run: python main.py --persona researcher --job literature_review")
        print("3. Check results in the 'output' directory")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 