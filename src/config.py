"""
Configuration module for Advanced Persona-Driven Document Intelligence v3.0
Contains persona knowledge bases, job patterns, and language detection patterns.
"""

# Enhanced persona-specific knowledge bases with multilingual keywords
PERSONA_KNOWLEDGE = {
    'researcher': {
        'keywords': {
            'en': {
                'methodology': 10, 'hypothesis': 9, 'experiment': 9, 'analysis': 8, 
                'findings': 8, 'conclusion': 8, 'literature review': 10, 'data': 7, 
                'statistics': 8, 'results': 8, 'discussion': 8, 'research question': 9, 
                'sampling': 7, 'variables': 7, 'correlation': 8, 'significance': 8,
                'peer review': 9, 'citation': 7, 'references': 7, 'abstract': 8,
                'background': 7, 'objectives': 8, 'limitations': 7, 'future work': 7
            },
            'es': {
                'metodología': 10, 'hipótesis': 9, 'experimento': 9, 'análisis': 8,
                'hallazgos': 8, 'conclusión': 8, 'revisión literaria': 10, 'datos': 7,
                'estadísticas': 8, 'resultados': 8, 'discusión': 8, 'pregunta investigación': 9,
                'muestreo': 7, 'variables': 7, 'correlación': 8, 'significancia': 8,
                'revisión pares': 9, 'cita': 7, 'referencias': 7, 'resumen': 8,
                'antecedentes': 7, 'objetivos': 8, 'limitaciones': 7, 'trabajo futuro': 7
            }
        },
        'sections': ['introduction', 'methodology', 'results', 'discussion', 'conclusion', 'abstract', 'literature review'],
        'focus_areas': ['research design', 'data analysis', 'statistical methods', 'findings interpretation'],
        'content_weights': {
            'technical_depth': 0.9,
            'methodology_focus': 0.8,
            'data_analysis': 0.9,
            'conceptual_framework': 0.7
        }
    },
    'student': {
        'keywords': {
            'en': {
                'concept': 9, 'definition': 9, 'example': 8, 'explanation': 8, 
                'theory': 8, 'principle': 8, 'formula': 8, 'step-by-step': 9, 
                'tutorial': 9, 'practice': 8, 'exercise': 8, 'summary': 8, 
                'key points': 9, 'learning objectives': 9, 'overview': 7,
                'introduction': 7, 'basics': 8, 'fundamentals': 8, 'guide': 8,
                'how to': 8, 'tips': 7, 'common mistakes': 8, 'review': 7
            }
        },
        'sections': ['introduction', 'overview', 'examples', 'summary', 'key concepts', 'practice problems'],
        'focus_areas': ['fundamental concepts', 'practical examples', 'step-by-step explanations'],
        'content_weights': {
            'clarity': 0.9,
            'examples': 0.8,
            'step_by_step': 0.8,
            'basics_focus': 0.9
        }
    },
    'analyst': {
        'keywords': {
            'en': {
                'trend': 9, 'pattern': 9, 'insight': 9, 'analysis': 8, 
                'comparison': 8, 'benchmark': 8, 'metric': 8, 'kpi': 9, 
                'performance': 8, 'evaluation': 8, 'assessment': 8, 'recommendation': 9, 
                'forecast': 9, 'projection': 8, 'market': 8, 'industry': 7,
                'competitive': 8, 'strategy': 8, 'optimization': 8, 'efficiency': 7,
                'roi': 9, 'growth': 8, 'opportunity': 8, 'risk': 8
            }
        },
        'sections': ['executive summary', 'analysis', 'findings', 'recommendations', 'conclusions'],
        'focus_areas': ['data interpretation', 'trend analysis', 'business insights'],
        'content_weights': {
            'data_driven': 0.9,
            'insights': 0.9,
            'actionable': 0.8,
            'strategic': 0.8
        }
    }
}

# Enhanced job-specific patterns with multilingual support
JOB_PATTERNS = {
    'literature_review': {
        'keywords': {
            'en': {
                'literature': 10, 'review': 9, 'research': 8, 'study': 8, 
                'paper': 7, 'publication': 7, 'methodology': 8, 'findings': 8, 
                'conclusion': 8, 'citation': 7, 'references': 7, 'background': 7,
                'existing work': 8, 'previous research': 8, 'gap': 8, 'contribution': 7
            }
        },
        'focus': 'comprehensive overview of existing research',
        'content_weights': {
            'comprehensive': 0.9,
            'analytical': 0.8,
            'synthesis': 0.8,
            'critical': 0.7
        }
    },
    'exam_preparation': {
        'keywords': {
            'en': {
                'concept': 9, 'definition': 9, 'formula': 8, 'example': 8, 
                'practice': 8, 'key points': 9, 'summary': 8, 'review': 8,
                'important': 8, 'essential': 8, 'core': 8, 'fundamental': 8,
                'test': 7, 'exam': 7, 'question': 7, 'answer': 7
            }
        },
        'focus': 'essential concepts and examples for testing',
        'content_weights': {
            'essential': 0.9,
            'clear': 0.8,
            'memorable': 0.7,
            'practical': 0.8
        }
    }
}

# Language detection patterns
LANGUAGE_PATTERNS = {
    'en': r'[a-zA-Z\s]+',
    'es': r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]+',
    'fr': r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s]+',
    'de': r'[a-zA-ZäöüßÄÖÜ\s]+',
    'hi': r'[\u0900-\u097F\s]+',  # Devanagari script
    'it': r'[a-zA-ZàèéìíîòóùÀÈÉÌÍÎÒÓÙ\s]+',
    'pt': r'[a-zA-ZàáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ\s]+'
}

# Section keywords for different languages
SECTION_KEYWORDS = {
    'en': ['introduction', 'methodology', 'results', 'discussion', 'conclusion'],
    'es': ['introducción', 'metodología', 'resultados', 'discusión', 'conclusión'],
    'fr': ['introduction', 'méthodologie', 'résultats', 'discussion', 'conclusion'],
    'de': ['einführung', 'methodik', 'ergebnisse', 'diskussion', 'schlussfolgerung'],
    'hi': ['परिचय', 'पद्धति', 'परिणाम', 'चर्चा', 'निष्कर्ष']
}

# Technical terms for different languages
TECHNICAL_TERMS = {
    'en': ['analysis', 'method', 'process', 'system', 'data', 'result'],
    'es': ['análisis', 'método', 'proceso', 'sistema', 'datos', 'resultado'],
    'fr': ['analyse', 'méthode', 'processus', 'système', 'données', 'résultat'],
    'de': ['analyse', 'methode', 'prozess', 'system', 'daten', 'ergebnis'],
    'hi': ['विश्लेषण', 'विधि', 'प्रक्रिया', 'सिस्टम', 'डेटा', 'परिणाम']
}

# Heading patterns for detection
HEADING_PATTERNS = [
    r'^\d+\.\s+[A-Z]',              # 1. Introduction
    r'^\d+\.\d+\s+[A-Z]',           # 1.1 Overview
    r'^[A-Z][A-Z\s]{3,}$',          # ALL CAPS headings
    r'^Chapter\s+\d+',              # Chapter 1
    r'^Section\s+\d+',              # Section 1
]

# Heading keywords
HEADING_KEYWORDS = {
    'introduction', 'overview', 'summary', 'conclusion', 'abstract',
    'background', 'methodology', 'results', 'discussion', 'references'
} 