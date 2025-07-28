#!/usr/bin/env python3
"""
Enhanced Adobe Hackathon: Advanced Persona-Driven Document Intelligence v3.0
"Connect What Matters — For the User Who Matters"

Advanced document analysis system with improved accuracy, better content analysis,
multilingual support, and sophisticated persona-specific algorithms for extracting 
and prioritizing relevant sections based on persona and job-to-be-done requirements.
"""

import json
import sys
import os
import re
import time
from typing import Dict, List, Tuple, Set, Optional
import fitz  # PyMuPDF
from collections import defaultdict, Counter
import argparse
from datetime import datetime
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

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

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class AdvancedPersonaIntelligence:
    def __init__(self):
        """Initialize the advanced persona intelligence system with multilingual support"""
        self.start_time = time.time()
        
        # Load spaCy models for multiple languages
        self.nlp_models = {}
        try:
            self.nlp_models['en'] = spacy.load("en_core_web_sm")
            print("Loaded English spaCy model", file=sys.stderr)
        except OSError:
            self.nlp_models['en'] = None
            print("Warning: English spaCy model not available", file=sys.stderr)
        
        # Try to load other language models
        for lang in ['es', 'fr', 'de', 'it', 'pt']:
            try:
                self.nlp_models[lang] = spacy.load(f"{lang}_core_news_sm")
                print(f"Loaded {lang} spaCy model", file=sys.stderr)
            except OSError:
                self.nlp_models[lang] = None
        
        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("Loaded multilingual sentence transformer", file=sys.stderr)
        except Exception as e:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded English sentence transformer as fallback", file=sys.stderr)
            except Exception as e:
                self.sentence_model = None
                print(f"Warning: Sentence transformer not available: {e}", file=sys.stderr)
        
        # Enhanced persona-specific knowledge bases with multilingual keywords
        self.persona_knowledge = {
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
                    },
                    'fr': {
                        'méthodologie': 10, 'hypothèse': 9, 'expérience': 9, 'analyse': 8,
                        'résultats': 8, 'conclusion': 8, 'revue littérature': 10, 'données': 7,
                        'statistiques': 8, 'résultats': 8, 'discussion': 8, 'question recherche': 9,
                        'échantillonnage': 7, 'variables': 7, 'corrélation': 8, 'significativité': 8,
                        'évaluation par pairs': 9, 'citation': 7, 'références': 7, 'résumé': 8,
                        'contexte': 7, 'objectifs': 8, 'limitations': 7, 'travaux futurs': 7
                    },
                    'de': {
                        'methodik': 10, 'hypothese': 9, 'experiment': 9, 'analyse': 8,
                        'ergebnisse': 8, 'schlussfolgerung': 8, 'literaturübersicht': 10, 'daten': 7,
                        'statistik': 8, 'ergebnisse': 8, 'diskussion': 8, 'forschungsfrage': 9,
                        'stichprobe': 7, 'variablen': 7, 'korrelation': 8, 'signifikanz': 8,
                        'peer review': 9, 'zitat': 7, 'referenzen': 7, 'zusammenfassung': 8,
                        'hintergrund': 7, 'ziele': 8, 'einschränkungen': 7, 'zukünftige arbeit': 7
                    },
                    'hi': {
                        'पद्धति': 10, 'परिकल्पना': 9, 'प्रयोग': 9, 'विश्लेषण': 8,
                        'निष्कर्ष': 8, 'निष्कर्ष': 8, 'साहित्य समीक्षा': 10, 'डेटा': 7,
                        'आंकड़े': 8, 'परिणाम': 8, 'चर्चा': 8, 'अनुसंधान प्रश्न': 9,
                        'नमूना': 7, 'चर': 7, 'सहसंबंध': 8, 'महत्व': 8,
                        'सहकर्मी समीक्षा': 9, 'उद्धरण': 7, 'संदर्भ': 7, 'सारांश': 8,
                        'पृष्ठभूमि': 7, 'उद्देश्य': 8, 'सीमाएं': 7, 'भविष्य का काम': 7
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
                    },
                    'es': {
                        'concepto': 9, 'definición': 9, 'ejemplo': 8, 'explicación': 8,
                        'teoría': 8, 'principio': 8, 'fórmula': 8, 'paso a paso': 9,
                        'tutorial': 9, 'práctica': 8, 'ejercicio': 8, 'resumen': 8,
                        'puntos clave': 9, 'objetivos aprendizaje': 9, 'descripción general': 7,
                        'introducción': 7, 'básicos': 8, 'fundamentos': 8, 'guía': 8,
                        'cómo': 8, 'consejos': 7, 'errores comunes': 8, 'revisión': 7
                    },
                    'fr': {
                        'concept': 9, 'définition': 9, 'exemple': 8, 'explication': 8,
                        'théorie': 8, 'principe': 8, 'formule': 8, 'étape par étape': 9,
                        'tutoriel': 9, 'pratique': 8, 'exercice': 8, 'résumé': 8,
                        'points clés': 9, 'objectifs apprentissage': 9, 'aperçu': 7,
                        'introduction': 7, 'bases': 8, 'fondamentaux': 8, 'guide': 8,
                        'comment': 8, 'conseils': 7, 'erreurs communes': 8, 'révision': 7
                    },
                    'de': {
                        'konzept': 9, 'definition': 9, 'beispiel': 8, 'erklärung': 8,
                        'theorie': 8, 'prinzip': 8, 'formel': 8, 'schritt für schritt': 9,
                        'tutorial': 9, 'übung': 8, 'aufgabe': 8, 'zusammenfassung': 8,
                        'schlüsselpunkte': 9, 'lernziele': 9, 'überblick': 7,
                        'einführung': 7, 'grundlagen': 8, 'grundlagen': 8, 'anleitung': 8,
                        'wie': 8, 'tipps': 7, 'häufige fehler': 8, 'wiederholung': 7
                    },
                    'hi': {
                        'अवधारणा': 9, 'परिभाषा': 9, 'उदाहरण': 8, 'व्याख्या': 8,
                        'सिद्धांत': 8, 'सिद्धांत': 8, 'सूत्र': 8, 'कदम दर कदम': 9,
                        'ट्यूटोरियल': 9, 'अभ्यास': 8, 'अभ्यास': 8, 'सारांश': 8,
                        'मुख्य बिंदु': 9, 'सीखने के उद्देश्य': 9, 'अवलोकन': 7,
                        'परिचय': 7, 'मूल बातें': 8, 'मूल सिद्धांत': 8, 'गाइड': 8,
                        'कैसे': 8, 'सुझाव': 7, 'सामान्य गलतियां': 8, 'समीक्षा': 7
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
                    },
                    'es': {
                        'tendencia': 9, 'patrón': 9, 'insight': 9, 'análisis': 8,
                        'comparación': 8, 'benchmark': 8, 'métrica': 8, 'kpi': 9,
                        'rendimiento': 8, 'evaluación': 8, 'evaluación': 8, 'recomendación': 9,
                        'pronóstico': 9, 'proyección': 8, 'mercado': 8, 'industria': 7,
                        'competitivo': 8, 'estrategia': 8, 'optimización': 8, 'eficiencia': 7,
                        'roi': 9, 'crecimiento': 8, 'oportunidad': 8, 'riesgo': 8
                    },
                    'fr': {
                        'tendance': 9, 'pattern': 9, 'insight': 9, 'analyse': 8,
                        'comparaison': 8, 'benchmark': 8, 'métrique': 8, 'kpi': 9,
                        'performance': 8, 'évaluation': 8, 'évaluation': 8, 'recommandation': 9,
                        'prévision': 9, 'projection': 8, 'marché': 8, 'industrie': 7,
                        'compétitif': 8, 'stratégie': 8, 'optimisation': 8, 'efficacité': 7,
                        'roi': 9, 'croissance': 8, 'opportunité': 8, 'risque': 8
                    },
                    'de': {
                        'trend': 9, 'muster': 9, 'insight': 9, 'analyse': 8,
                        'vergleich': 8, 'benchmark': 8, 'metrik': 8, 'kpi': 9,
                        'leistung': 8, 'bewertung': 8, 'bewertung': 8, 'empfehlung': 9,
                        'vorhersage': 9, 'prognose': 8, 'markt': 8, 'branche': 7,
                        'wettbewerb': 8, 'strategie': 8, 'optimierung': 8, 'effizienz': 7,
                        'roi': 9, 'wachstum': 8, 'chance': 8, 'risiko': 8
                    },
                    'hi': {
                        'प्रवृत्ति': 9, 'पैटर्न': 9, 'ज्ञान': 9, 'विश्लेषण': 8,
                        'तुलना': 8, 'बेंचमार्क': 8, 'मीट्रिक': 8, 'कीपीआई': 9,
                        'प्रदर्शन': 8, 'आंकलन': 8, 'मूल्यांकन': 8, 'सुझाव': 9,
                        'अनुमान': 9, 'प्रक्षेपण': 8, 'बाजार': 8, 'वातावरण': 7,
                        'प्रतिस्पर्धी': 8, 'रणनीति': 8, 'अनुकूलन': 8, 'दक्षता': 7,
                        'रोइ': 9, 'विकास': 8, 'अवसर': 8, 'जोखिम': 8
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
            },
            'manager': {
                'keywords': {
                    'en': {
                        'strategy': 9, 'planning': 8, 'execution': 8, 'leadership': 8, 
                        'decision': 9, 'overview': 7, 'summary': 8, 'roi': 9, 
                        'budget': 8, 'timeline': 8, 'objectives': 8, 'goals': 8, 
                        'performance': 8, 'stakeholder': 8, 'team': 7, 'project': 8,
                        'implementation': 8, 'monitoring': 7, 'control': 7, 'success': 7,
                        'challenge': 7, 'solution': 8, 'priority': 8, 'resource': 7
                    },
                    'es': {
                        'estrategia': 9, 'planificación': 8, 'ejecución': 8, 'liderazgo': 8,
                        'decisión': 9, 'resumen': 7, 'resumen': 8, 'roi': 9,
                        'presupuesto': 8, 'cronograma': 8, 'objetivos': 8, 'objetivos': 8,
                        'rendimiento': 8, 'inversionista': 8, 'equipo': 7, 'proyecto': 8,
                        'implementación': 8, 'monitoreo': 7, 'control': 7, 'éxito': 7,
                        'reto': 7, 'solución': 8, 'prioridad': 8, 'recurso': 7
                    },
                    'fr': {
                        'stratégie': 9, 'planification': 8, 'exécution': 8, 'leadership': 8,
                        'décision': 9, 'aperçu': 7, 'résumé': 8, 'roi': 9,
                        'budget': 8, 'timeline': 8, 'objectifs': 8, 'objectifs': 8,
                        'performance': 8, 'partenaire': 8, 'équipe': 7, 'projet': 8,
                        'implémentation': 8, 'surveillance': 7, 'contrôle': 7, 'succès': 7,
                        'défi': 7, 'solution': 8, 'priorité': 8, 'ressource': 7
                    },
                    'de': {
                        'strategie': 9, 'planung': 8, 'ausführung': 8, 'führung': 8,
                        'entscheidung': 9, 'übersicht': 7, 'zusammenfassung': 8, 'roi': 9,
                        'budget': 8, 'zeitplan': 8, 'ziele': 8, 'ziele': 8,
                        'leistung': 8, 'stakeholder': 8, 'team': 7, 'projekt': 8,
                        'implementierung': 8, 'überwachung': 7, 'kontrolle': 7, 'erfolg': 7,
                        'herausforderung': 7, 'lösung': 8, 'priorität': 8, 'ressource': 7
                    },
                    'hi': {
                        'रणनीति': 9, 'योजना': 8, 'निर्देश': 8, 'नेतृत्व': 8,
                        'निर्णायक': 9, 'अवलोकन': 7, 'सारांश': 8, 'रोइ': 9,
                        'बजट': 8, 'टाइमलाइन': 8, 'उद्देश्य': 8, 'लक्ष्य': 8,
                        'प्रदर्शन': 8, 'स्टेकहोल्डर': 8, 'टीम': 7, 'परियोजना': 8,
                        'निर्माण': 8, 'निगरानी': 7, 'नियंत्रण': 7, 'सफलता': 7,
                        'चुनौती': 7, 'समाधान': 8, 'उपाध्यक्षता': 8, 'संसाधन': 7
                    }
                },
                'sections': ['executive summary', 'strategy', 'planning', 'implementation', 'conclusions'],
                'focus_areas': ['strategic overview', 'decision points', 'implementation planning'],
                'content_weights': {
                    'strategic': 0.9,
                    'executive': 0.8,
                    'actionable': 0.8,
                    'overview': 0.7
                }
            },
            'developer': {
                'keywords': {
                    'en': {
                        'implementation': 9, 'code': 9, 'architecture': 9, 'algorithm': 8, 
                        'api': 8, 'framework': 8, 'deployment': 8, 'testing': 8, 
                        'debugging': 8, 'optimization': 8, 'performance': 8, 'security': 8,
                        'database': 7, 'frontend': 7, 'backend': 7, 'integration': 8,
                        'configuration': 7, 'setup': 7, 'installation': 7, 'troubleshooting': 8,
                        'best practices': 8, 'design pattern': 8, 'scalability': 7, 'maintenance': 7
                    },
                    'es': {
                        'implementación': 9, 'código': 9, 'arquitectura': 9, 'algoritmo': 8,
                        'api': 8, 'framework': 8, 'implementación': 8, 'pruebas': 8,
                        'depuración': 8, 'optimización': 8, 'rendimiento': 8, 'seguridad': 8,
                        'base de datos': 7, 'frontend': 7, 'backend': 7, 'integración': 8,
                        'configuración': 7, 'instalación': 7, 'instalación': 7, 'solución de problemas': 8,
                        'mejores prácticas': 8, 'patrón de diseño': 8, 'escalabilidad': 7, 'mantenimiento': 7
                    },
                    'fr': {
                        'implémentation': 9, 'code': 9, 'architecture': 9, 'algorithme': 8,
                        'api': 8, 'framework': 8, 'déploiement': 8, 'test': 8,
                        'debugging': 8, 'optimisation': 8, 'performance': 8, 'sécurité': 8,
                        'base de données': 7, 'frontend': 7, 'backend': 7, 'intégration': 8,
                        'configuration': 7, 'installation': 7, 'installation': 7, 'résolution de problèmes': 8,
                        'meilleures pratiques': 8, 'patron de conception': 8, 'scalabilité': 7, 'maintenance': 7
                    },
                    'de': {
                        'implementierung': 9, 'code': 9, 'architektur': 9, 'algorithmus': 8,
                        'api': 8, 'framework': 8, 'deployment': 8, 'test': 8,
                        'debugging': 8, 'optimierung': 8, 'leistung': 8, 'sicherheit': 8,
                        'datenbank': 7, 'frontend': 7, 'backend': 7, 'integration': 8,
                        'konfiguration': 7, 'einrichtung': 7, 'installation': 7, 'troubleshooting': 8,
                        'best practices': 8, 'design pattern': 8, 'skalierbarkeit': 7, 'wartung': 7
                    },
                    'hi': {
                        'निर्माण': 9, 'कोड': 9, 'संरचना': 9, 'एल्गोरिदम': 8,
                        'एपीआई': 8, 'फ्रेमवर्क': 8, 'डिप्लोईमेंट': 8, 'टेस्टिंग': 8,
                        'डीबगिंग': 8, 'ऑप्टिमाइजेशन': 8, 'प्रदर्शन': 8, 'सुरक्षा': 8,
                        'डेटाबेस': 7, 'फ्रंटएंड': 7, 'बैकएंड': 7, 'इंटीग्रेशन': 8,
                        'कॉन्फ़िगरेशन': 7, 'इंस्टॉलेशन': 7, 'इंस्टॉलेशन': 7, 'ट्यूबलेशन': 8,
                        'सर्वोपद्र्भव': 8, 'डिज़ाइन पैटर्न': 8, 'स्केलेबिलिटी': 7, 'मेंटेनेंस': 7
                    }
                },
                'sections': ['implementation', 'architecture', 'code examples', 'deployment', 'troubleshooting'],
                'focus_areas': ['technical implementation', 'code structure', 'system architecture'],
                'content_weights': {
                    'technical': 0.9,
                    'implementation': 0.9,
                    'code_focus': 0.8,
                    'practical': 0.8
                }
            },
            'journalist': {
                'keywords': {
                    'en': {
                        'news': 9, 'story': 8, 'event': 8, 'interview': 8, 
                        'source': 8, 'fact': 8, 'quote': 8, 'timeline': 7, 
                        'background': 7, 'context': 7, 'impact': 8, 'reaction': 7, 
                        'statement': 7, 'announcement': 7, 'development': 7, 'update': 7,
                        'investigation': 8, 'report': 8, 'coverage': 7, 'breaking': 8,
                        'exclusive': 8, 'analysis': 7, 'perspective': 7, 'opinion': 7
                    },
                    'es': {
                        'noticias': 9, 'historia': 8, 'evento': 8, 'entrevista': 8,
                        'fuente': 8, 'hecho': 8, 'cita': 8, 'cronología': 7,
                        'antecedentes': 7, 'contexto': 7, 'impacto': 8, 'reacción': 7,
                        'declaración': 7, 'anuncio': 7, 'desarrollo': 7, 'actualización': 7,
                        'investigación': 8, 'informe': 8, 'cobertura': 7, 'rompiendo': 8,
                        'exclusivo': 8, 'análisis': 7, 'perspectiva': 7, 'opinión': 7
                    },
                    'fr': {
                        'news': 9, 'story': 8, 'event': 8, 'interview': 8,
                        'source': 8, 'fact': 8, 'quote': 8, 'timeline': 7,
                        'background': 7, 'context': 7, 'impact': 8, 'reaction': 7,
                        'statement': 7, 'announcement': 7, 'development': 7, 'update': 7,
                        'investigation': 8, 'report': 8, 'coverage': 7, 'breaking': 8,
                        'exclusive': 8, 'analysis': 7, 'perspective': 7, 'opinion': 7
                    },
                    'de': {
                        'news': 9, 'story': 8, 'event': 8, 'interview': 8,
                        'source': 8, 'fact': 8, 'quote': 8, 'timeline': 7,
                        'background': 7, 'context': 7, 'impact': 8, 'reaction': 7,
                        'statement': 7, 'announcement': 7, 'entwicklung': 7, 'aktualisierung': 7,
                        'aufklärung': 8, 'bericht': 8, 'deckung': 7, 'brechend': 8,
                        'exklusiv': 8, 'analyse': 7, 'perspektive': 7, 'meinung': 7
                    },
                    'hi': {
                        'समाचार': 9, 'कहानी': 8, 'घटना': 8, 'साक्षात्कार': 8,
                        'स्रोत': 8, 'तथ्य': 8, 'उद्धरण': 8, 'क्रोनोलॉजी': 7,
                        'पृष्ठभूमि': 7, 'संदर्भ': 7, 'प्रभाव': 8, 'प्रतिक्रिया': 7,
                        'घोषणा': 7, 'घोषणा': 7, 'विकास': 7, 'अद्यतन': 7,
                        'अध्ययन': 8, 'रिपोर्ट': 8, 'कवर': 7, 'टूटना': 8,
                        'अक्सेसोरी': 8, 'विश्लेषण': 7, 'दृष्टिकोण': 7, 'राय': 7
                    }
                },
                'sections': ['headlines', 'lead', 'body', 'quotes', 'background', 'conclusion'],
                'focus_areas': ['key facts', 'quotes', 'timeline', 'context'],
                'content_weights': {
                    'factual': 0.9,
                    'timely': 0.8,
                    'objective': 0.8,
                    'engaging': 0.7
                }
            },
            'entrepreneur': {
                'keywords': {
                    'en': {
                        'opportunity': 9, 'market': 9, 'business model': 9, 'revenue': 9, 
                        'growth': 9, 'strategy': 8, 'competition': 8, 'investment': 9, 
                        'scaling': 8, 'innovation': 8, 'risk': 8, 'pitch': 8,
                        'startup': 8, 'funding': 8, 'customer': 8, 'product': 8,
                        'go-to-market': 8, 'monetization': 8, 'partnership': 7, 'acquisition': 7,
                        'exit strategy': 8, 'competitive advantage': 8, 'market fit': 8, 'traction': 8
                    },
                    'es': {
                        'oportunidad': 9, 'mercado': 9, 'modelo de negocio': 9, 'ingresos': 9,
                        'crecimiento': 9, 'estrategia': 8, 'competencia': 8, 'inversión': 9,
                        'escalado': 8, 'innovación': 8, 'riesgo': 8, 'pitch': 8,
                        'startup': 8, 'financiación': 8, 'cliente': 8, 'producto': 8,
                        'mercado': 8, 'monetización': 8, 'sociedad de colaboración': 7, 'adquisición': 7,
                        'estrategia de salida': 8, 'ventaja competitiva': 8, 'ajuste de mercado': 8, 'tirón': 8
                    },
                    'fr': {
                        'opportunité': 9, 'marché': 9, 'modèle d\'affaires': 9, 'revenu': 9,
                        'croissance': 9, 'stratégie': 8, 'compétition': 8, 'investissement': 9,
                        'échelle': 8, 'innovation': 8, 'risque': 8, 'pitch': 8,
                        'startup': 8, 'financement': 8, 'client': 8, 'produit': 8,
                        'aller-à-la-clientèle': 8, 'monétisation': 8, 'partenariat': 7, 'acquisition': 7,
                        'stratégie d\'exit': 8, 'avantage concurrentiel': 8, 'ajustement de marché': 8, 'traction': 8
                    },
                    'de': {
                        'chance': 9, 'markt': 9, 'geschäftsmodell': 9, 'umsatz': 9,
                        'wachstum': 9, 'strategie': 8, 'wettbewerb': 8, 'investition': 9,
                        'skalierung': 8, 'innovation': 8, 'risiko': 8, 'pitch': 8,
                        'startup': 8, 'finanzierung': 8, 'kunde': 8, 'produkt': 8,
                        'gehen-zum-kunden': 8, 'monetisierung': 8, 'partnerschaft': 7, 'einlage': 7,
                        'ausstieg-strategie': 8, 'wettbewerbsfähigkeit': 8, 'marktpassung': 8, 'zugkraft': 8
                    },
                    'hi': {
                        'अवसर': 9, 'बाजार': 9, 'व्यवसाय मॉडल': 9, 'राजस्व': 9,
                        'विकास': 9, 'रणनीति': 8, 'प्रतिस्पर्धा': 8, 'निवेश': 9,
                        'स्केलिंग': 8, 'नवाचार': 8, 'जोखिम': 8, 'पिच': 8,
                        'स्टार्टअप': 8, 'फाइनेंसिंग': 8, 'ग्राहक': 8, 'उत्पाद': 8,
                        'जाने-जाने-ग्राहक': 8, 'मोनेटाइजेशन': 8, 'साझेदारी': 7, 'खरीद': 7,
                        'निर्गम रणनीति': 8, 'प्रतिस्पर्धीता': 8, 'बाजार फिट': 8, 'ट्रैक्शन': 8
                    }
                },
                'sections': ['executive summary', 'market analysis', 'business model', 'strategy', 'financials'],
                'focus_areas': ['market opportunities', 'business strategy', 'financial projections'],
                'content_weights': {
                    'opportunity': 0.9,
                    'market_focus': 0.9,
                    'business': 0.8,
                    'growth': 0.8
                }
            }
        }
        
        # Enhanced job-specific patterns with multilingual support
        self.job_patterns = {
            'literature_review': {
                'keywords': {
                    'en': {
                        'literature': 10, 'review': 9, 'research': 8, 'study': 8, 
                        'paper': 7, 'publication': 7, 'methodology': 8, 'findings': 8, 
                        'conclusion': 8, 'citation': 7, 'references': 7, 'background': 7,
                        'existing work': 8, 'previous research': 8, 'gap': 8, 'contribution': 7
                    },
                    'es': {
                        'literatura': 10, 'revisión': 9, 'investigación': 8, 'estudio': 8,
                        'artículo': 7, 'publicación': 7, 'metodología': 8, 'hallazgos': 8,
                        'conclusión': 8, 'cita': 7, 'referencias': 7, 'antecedentes': 7,
                        'trabajo existente': 8, 'investigación previa': 8, 'brecha': 8, 'contribución': 7
                    },
                    'fr': {
                        'littérature': 10, 'revue': 9, 'recherche': 8, 'étude': 8,
                        'article': 7, 'publication': 7, 'méthodologie': 8, 'résultats': 8,
                        'conclusion': 8, 'citation': 7, 'références': 7, 'contexte': 7,
                        'travail existant': 8, 'recherche précédente': 8, 'écart': 8, 'contribution': 7
                    },
                    'de': {
                        'literatur': 10, 'übersicht': 9, 'forschung': 8, 'studie': 8,
                        'artikel': 7, 'veröffentlichung': 7, 'methodik': 8, 'ergebnisse': 8,
                        'schlussfolgerung': 8, 'zitat': 7, 'referenzen': 7, 'hintergrund': 7,
                        'bestehende arbeit': 8, 'vorherige forschung': 8, 'lücke': 8, 'beitrag': 7
                    },
                    'hi': {
                        'साहित्य': 10, 'समीक्षा': 9, 'अनुसंधान': 8, 'अध्ययन': 8,
                        'पेपर': 7, 'प्रकाशन': 7, 'पद्धति': 8, 'निष्कर्ष': 8,
                        'निष्कर्ष': 8, 'उद्धरण': 7, 'संदर्भ': 7, 'पृष्ठभूमि': 7,
                        'मौजूदा काम': 8, 'पिछला अनुसंधान': 8, 'अंतर': 8, 'योगदान': 7
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
                    },
                    'es': {
                        'concepto': 9, 'definición': 9, 'fórmula': 8, 'ejemplo': 8,
                        'práctica': 8, 'puntos clave': 9, 'resumen': 8, 'revisión': 8,
                        'importante': 8, 'esencial': 8, 'núcleo': 8, 'fundamental': 8,
                        'prueba': 7, 'examen': 7, 'pregunta': 7, 'respuesta': 7
                    },
                    'fr': {
                        'concept': 9, 'définition': 9, 'exemple': 8, 'explication': 8,
                        'pratique': 8, 'points clés': 9, 'résumé': 8, 'révision': 8,
                        'important': 8, 'essentiel': 8, 'cœur': 8, 'fondamental': 8,
                        'test': 7, 'examen': 7, 'question': 7, 'réponse': 7
                    },
                    'de': {
                        'konzept': 9, 'definition': 9, 'formel': 8, 'beispiel': 8,
                        'übung': 8, 'schlüsselpunkte': 9, 'zusammenfassung': 8, 'wiederholung': 7,
                        'wichtig': 8, 'essentiel': 8, 'kern': 8, 'fundamental': 8,
                        'test': 7, 'prüfung': 7, 'frage': 7, 'antwort': 7
                    },
                    'hi': {
                        'अवधारणा': 9, 'परिभाषा': 9, 'सूत्र': 8, 'उदाहरण': 8,
                        'अभ्यास': 8, 'मुख्य बिंदु': 9, 'सारांश': 8, 'समीक्षा': 8,
                        'महत्वपूर्ण': 8, 'अत्यधिक': 8, 'केंद्रीय': 8, 'मूलभूत': 8,
                        'परीक्षा': 7, 'परीक्षा': 7, 'प्रश्न': 7, 'उत्तर': 7
                    }
                },
                'focus': 'essential concepts and examples for testing',
                'content_weights': {
                    'essential': 0.9,
                    'clear': 0.8,
                    'memorable': 0.7,
                    'practical': 0.8
                }
            },
            'data_analysis': {
                'keywords': {
                    'en': {
                        'data': 9, 'analysis': 9, 'trend': 8, 'pattern': 8, 
                        'statistics': 8, 'insight': 8, 'finding': 8, 'correlation': 8,
                        'regression': 7, 'clustering': 7, 'visualization': 7, 'chart': 7,
                        'graph': 7, 'metric': 8, 'kpi': 8, 'performance': 7
                    },
                    'es': {
                        'datos': 9, 'análisis': 9, 'tendencia': 8, 'patrón': 8,
                        'estadísticas': 8, 'insight': 8, 'hallazgo': 8, 'correlación': 8,
                        'regresión': 7, 'agrupación': 7, 'visualización': 7, 'gráfico': 7,
                        'grafico': 7, 'métrica': 8, 'kpi': 8, 'rendimiento': 7
                    },
                    'fr': {
                        'données': 9, 'analyse': 9, 'tendance': 8, 'pattern': 8,
                        'statistiques': 8, 'insight': 8, 'découverte': 8, 'corrélation': 8,
                        'régression': 7, 'clustering': 7, 'visualisation': 7, 'graphique': 7,
                        'graphe': 7, 'métrique': 8, 'kpi': 8, 'performance': 7
                    },
                    'de': {
                        'daten': 9, 'analyse': 9, 'trend': 8, 'muster': 8,
                        'statistik': 8, 'insight': 8, 'findung': 8, 'korrelation': 8,
                        'regression': 7, 'clustering': 7, 'visualisierung': 7, 'grafik': 7,
                        'graph': 7, 'metrik': 8, 'kpi': 8, 'leistung': 7
                    },
                    'hi': {
                        'डेटा': 9, 'विश्लेषण': 9, 'प्रवृत्ति': 8, 'पैटर्न': 8,
                        'आंकड़े': 8, 'ज्ञान': 8, 'खोज': 8, 'सहसंबंध': 8,
                        'प्रतिलोम': 7, 'समूहन': 7, 'विज़ुअलाइज़ेशन': 7, 'चार्ट': 7,
                        'ग्राफ': 7, 'मीट्रिक': 8, 'कीपीआई': 8, 'प्रदर्शन': 7
                    }
                },
                'focus': 'data-driven insights and analytical findings',
                'content_weights': {
                    'data_driven': 0.9,
                    'analytical': 0.9,
                    'insightful': 0.8,
                    'quantitative': 0.8
                }
            },
            'decision_making': {
                'keywords': {
                    'en': {
                        'decision': 9, 'choice': 8, 'option': 8, 'alternative': 8, 
                        'pros': 8, 'cons': 8, 'recommendation': 9, 'strategy': 8,
                        'risk': 8, 'benefit': 8, 'impact': 8, 'outcome': 8,
                        'evaluation': 8, 'assessment': 8, 'criteria': 8, 'priority': 8
                    },
                    'es': {
                        'decisión': 9, 'opción': 8, 'alternativa': 8, 'ventaja': 8,
                        'desventaja': 8, 'recomendación': 9, 'estrategia': 8,
                        'riesgo': 8, 'beneficio': 8, 'impacto': 8, 'resultado': 8,
                        'evaluación': 8, 'evaluación': 8, 'criterio': 8, 'prioridad': 8
                    },
                    'fr': {
                        'décision': 9, 'choix': 8, 'option': 8, 'alternative': 8,
                        'avantage': 8, 'inconvénient': 8, 'recommandation': 9, 'stratégie': 8,
                        'risque': 8, 'bénéfice': 8, 'impact': 8, 'issue': 8,
                        'évaluation': 8, 'évaluation': 8, 'critère': 8, 'priorité': 8
                    },
                    'de': {
                        'entscheidung': 9, 'auswahl': 8, 'option': 8, 'alternative': 8,
                        'vorteil': 8, 'nachteil': 8, 'empfehlung': 9, 'strategie': 8,
                        'risiko': 8, 'vorteil': 8, 'einfluss': 8, 'ausgang': 8,
                        'bewertung': 8, 'bewertung': 8, 'kriterium': 8, 'priorität': 8
                    },
                    'hi': {
                        'निर्णायक': 9, 'विकल्प': 8, 'विकल्प': 8, 'वैकल्पिक': 8,
                        'लाभ': 8, 'हानि': 8, 'सुझाव': 9, 'रणनीति': 8,
                        'जोखिम': 8, 'लाभ': 8, 'प्रभाव': 8, 'परिणाम': 8,
                        'आंकलन': 8, 'मूल्यांकन': 8, 'मानक': 8, 'प्राथमिकता': 8
                    }
                },
                'focus': 'decision support and strategic choices',
                'content_weights': {
                    'strategic': 0.9,
                    'comparative': 0.8,
                    'actionable': 0.8,
                    'comprehensive': 0.7
                }
            },
            'implementation': {
                'keywords': {
                    'en': {
                        'implementation': 9, 'execution': 8, 'deployment': 8, 'setup': 8, 
                        'configuration': 8, 'installation': 8, 'integration': 8, 'testing': 8,
                        'rollout': 7, 'migration': 7, 'upgrade': 7, 'maintenance': 7,
                        'troubleshooting': 8, 'best practice': 8, 'guideline': 7, 'procedure': 7
                    },
                    'es': {
                        'implementación': 9, 'ejecución': 8, 'implementación': 8, 'configuración': 8,
                        'instalación': 8, 'integración': 8, 'pruebas': 8,
                        'rollout': 7, 'migración': 7, 'actualización': 7, 'mantenimiento': 7,
                        'solución de problemas': 8, 'mejores prácticas': 8, 'guía': 7, 'procedimiento': 7
                    },
                    'fr': {
                        'implémentation': 9, 'exécution': 8, 'déploiement': 8, 'configuration': 8,
                        'installation': 8, 'intégration': 8, 'test': 8,
                        'rollout': 7, 'migration': 7, 'upgrade': 7, 'maintenance': 7,
                        'résolution de problèmes': 8, 'meilleures pratiques': 8, 'ligne directrice': 7, 'procédure': 7
                    },
                    'de': {
                        'implementierung': 9, 'ausführung': 8, 'deployment': 8, 'konfiguration': 8,
                        'installation': 8, 'integration': 8, 'test': 8,
                        'rollout': 7, 'migration': 7, 'upgrade': 7, 'wartung': 7,
                        'troubleshooting': 8, 'best practice': 8, 'anleitung': 7, 'verfahren': 7
                    },
                    'hi': {
                        'निर्माण': 9, 'निर्देश': 8, 'डिप्लोईमेंट': 8, 'सेटअप': 8,
                        'कॉन्फ़िगरेशन': 8, 'इंस्टॉलेशन': 8, 'इंटीग्रेशन': 8, 'टेस्टिंग': 8,
                        'रोलआउट': 7, 'मिग्रेशन': 7, 'अपग्रेड': 7, 'मेंटेनेंस': 7,
                        'सॉल्यूशन ऑफ प्रॉब्लेम्स': 8, 'बेस्ट प्रैक्टिस': 8, 'लाइन डायरेक्शन': 7, 'प्रोसेस': 7
                    }
                },
                'focus': 'practical implementation and execution',
                'content_weights': {
                    'practical': 0.9,
                    'step_by_step': 0.8,
                    'technical': 0.8,
                    'actionable': 0.8
                }
            },
            'news_reporting': {
                'keywords': {
                    'en': {
                        'news': 9, 'report': 8, 'story': 8, 'event': 8, 
                        'announcement': 8, 'development': 7, 'update': 7, 'breaking': 8,
                        'exclusive': 8, 'interview': 7, 'quote': 7, 'source': 7,
                        'timeline': 7, 'background': 7, 'context': 7, 'impact': 7
                    },
                    'es': {
                        'noticias': 9, 'reporte': 8, 'historia': 8, 'evento': 8,
                        'anuncio': 8, 'desarrollo': 7, 'actualización': 7, 'rompiendo': 8,
                        'exclusivo': 8, 'entrevista': 7, 'cita': 7, 'fuente': 7,
                        'cronología': 7, 'antecedentes': 7, 'contexto': 7, 'impacto': 7
                    },
                    'fr': {
                        'news': 9, 'report': 8, 'story': 8, 'event': 8,
                        'announcement': 8, 'development': 7, 'update': 7, 'breaking': 8,
                        'exclusive': 8, 'interview': 7, 'quote': 7, 'source': 7,
                        'timeline': 7, 'background': 7, 'context': 7, 'impact': 7
                    },
                    'de': {
                        'news': 9, 'bericht': 8, 'geschichte': 8, 'ereignis': 8,
                        'ankündigung': 8, 'entwicklung': 7, 'aktualisierung': 7, 'brechend': 8,
                        'exklusiv': 8, 'interview': 7, 'zitat': 7, 'quelle': 7,
                        'zeitachse': 7, 'hintergrund': 7, 'kontext': 7, 'einfluss': 7
                    },
                    'hi': {
                        'समाचार': 9, 'रिपोर्ट': 8, 'कहानी': 8, 'घटना': 8,
                        'घोषणा': 8, 'विकास': 7, 'अद्यतन': 7, 'टूटना': 8,
                        'अक्सेसोरी': 8, 'साक्षात्कार': 7, 'उद्धरण': 7, 'स्रोत': 7,
                        'क्रोनोलॉजी': 7, 'टाइमलाइन': 7, 'पृष्ठभूमि': 7, 'संदर्भ': 7,
                        'प्रभाव': 7
                    }
                },
                'focus': 'newsworthy information and current events',
                'content_weights': {
                    'timely': 0.9,
                    'factual': 0.8,
                    'objective': 0.8,
                    'engaging': 0.7
                }
            },
            'business_planning': {
                'keywords': {
                    'en': {
                        'business': 9, 'plan': 9, 'strategy': 8, 'market': 8, 
                        'revenue': 8, 'growth': 8, 'investment': 8, 'financial': 8,
                        'model': 8, 'opportunity': 8, 'competition': 7, 'risk': 7,
                        'milestone': 7, 'timeline': 7, 'resource': 7, 'success': 7
                    },
                    'es': {
                        'negocio': 9, 'plan': 9, 'estrategia': 8, 'mercado': 8,
                        'ingresos': 8, 'crecimiento': 8, 'inversión': 8, 'financiero': 8,
                        'modelo': 8, 'oportunidad': 8, 'competencia': 7, 'riesgo': 7,
                        'hitos': 7, 'cronograma': 7, 'recurso': 7, 'éxito': 7
                    },
                    'fr': {
                        'entreprise': 9, 'planification': 9, 'stratégie': 8, 'marché': 8,
                        'revenu': 8, 'croissance': 8, 'investissement': 8, 'financier': 8,
                        'modèle': 8, 'opportunité': 8, 'compétition': 7, 'risque': 7,
                        'milestone': 7, 'timeline': 7, 'ressource': 7, 'succès': 7
                    },
                    'de': {
                        'unternehmen': 9, 'planung': 9, 'strategie': 8, 'markt': 8,
                        'umsatz': 8, 'wachstum': 8, 'investition': 8, 'finanzierung': 8,
                        'modell': 8, 'chance': 8, 'wettbewerb': 7, 'risiko': 7,
                        'milestone': 7, 'zeitplan': 7, 'ressource': 7, 'erfolg': 7
                    },
                    'hi': {
                        'व्यवसाय': 9, 'योजना': 9, 'रणनीति': 8, 'बाजार': 8,
                        'राजस्व': 8, 'विकास': 8, 'निवेश': 8, 'आंकलनीय': 8,
                        'मॉडल': 8, 'अवसर': 8, 'प्रतिस्पर्धा': 7, 'जोखिम': 7,
                        'मिलस्टोन': 7, 'टाइमलाइन': 7, 'संसाधन': 7, 'सफलता': 7
                    }
                },
                'focus': 'business strategy and planning',
                'content_weights': {
                    'strategic': 0.9,
                    'business_focused': 0.9,
                    'comprehensive': 0.8,
                    'actionable': 0.8
                }
            }
        }
        
        # Language detection patterns
        self.language_patterns = {
            'en': r'[a-zA-Z\s]+',
            'es': r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]+',
            'fr': r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s]+',
            'de': r'[a-zA-ZäöüßÄÖÜ\s]+',
            'hi': r'[\u0900-\u097F\s]+',  # Devanagari script
            'it': r'[a-zA-ZàèéìíîòóùÀÈÉÌÍÎÒÓÙ\s]+',
            'pt': r'[a-zA-ZàáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ\s]+'
        }
        
        # Language detection patterns
        self.language_patterns = {
            'en': r'[a-zA-Z\s]+',
            'es': r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]+',
            'fr': r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ\s]+',
            'de': r'[a-zA-ZäöüßÄÖÜ\s]+',
            'hi': r'[\u0900-\u097F\s]+',  # Devanagari script
            'it': r'[a-zA-ZàèéìíîòóùÀÈÉÌÍÎÒÓÙ\s]+',
            'pt': r'[a-zA-ZàáâãçéêíóôõúÀÁÂÃÇÉÊÍÓÔÕÚ\s]+'
        }

    def check_time_limit(self, stage: str) -> bool:
        """Check if processing time limit is exceeded"""
        elapsed = time.time() - self.start_time
        if elapsed > 300:  # 5 minutes limit
            print(f"Time limit exceeded at {stage}", file=sys.stderr)
            return False
        return True

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text"""
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
        text_lower = text.lower()
        lang_scores = {}
        
        for lang, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                lang_scores[lang] = matches / len(text_lower)
        
        if lang_scores:
            return max(lang_scores, key=lang_scores.get)
        
        return 'en'  # Default to English

    def get_keywords_for_language(self, persona: str, job: str, language: str) -> Dict[str, int]:
        """Get keywords for the specified persona, job, and language"""
        persona_keywords = self.persona_knowledge.get(persona, {}).get('keywords', {})
        job_keywords = self.job_patterns.get(job, {}).get('keywords', {})
        
        # Get language-specific keywords
        persona_lang_keywords = persona_keywords.get(language, persona_keywords.get('en', {}))
        job_lang_keywords = job_keywords.get(language, job_keywords.get('en', {}))
        
        # Combine keywords
        combined_keywords = {}
        combined_keywords.update(persona_lang_keywords)
        combined_keywords.update(job_lang_keywords)
        
        return combined_keywords

    def calculate_semantic_relevance_multilingual(self, text: str, persona: str, job: str) -> float:
        """Calculate semantic relevance with multilingual support"""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Detect language
        detected_lang = self.detect_language(text)
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
        structure_score = 0
        
        # Check for relevant sections in multiple languages
        section_keywords = {
            'en': ['introduction', 'methodology', 'results', 'discussion', 'conclusion'],
            'es': ['introducción', 'metodología', 'resultados', 'discusión', 'conclusión'],
            'fr': ['introduction', 'méthodologie', 'résultats', 'discussion', 'conclusion'],
            'de': ['einführung', 'methodik', 'ergebnisse', 'diskussion', 'schlussfolgerung'],
            'hi': ['परिचय', 'पद्धति', 'परिणाम', 'चर्चा', 'निष्कर्ष']
        }
        
        lang_sections = section_keywords.get(detected_lang, section_keywords['en'])
        for section in lang_sections:
            if section in text_lower:
                structure_score += 5
        
        relevance_score += min(structure_score, 20)
        
        # 4. Text quality analysis (10% weight)
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
            technical_terms = {
                'en': ['analysis', 'method', 'process', 'system', 'data', 'result'],
                'es': ['análisis', 'método', 'proceso', 'sistema', 'datos', 'resultado'],
                'fr': ['analyse', 'méthode', 'processus', 'système', 'données', 'résultat'],
                'de': ['analyse', 'methode', 'prozess', 'system', 'daten', 'ergebnis'],
                'hi': ['विश्लेषण', 'विधि', 'प्रक्रिया', 'सिस्टम', 'डेटा', 'परिणाम']
            }
            lang_terms = technical_terms.get(detected_lang, technical_terms['en'])
            if any(term in text_lower for term in lang_terms):
                quality_score += 2
        
        relevance_score += quality_score
        
        return min(100, max(0, relevance_score))

    def extract_document_content_multilingual(self, pdf_path: str) -> Dict:
        """Extract document content with multilingual support"""
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
                if not self.check_time_limit(f"page {page_num}"):
                    break
                
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
                                detected_lang = self.detect_language(line_text.strip())
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

    def extract_relevant_sections_multilingual(self, content: Dict, persona: str, job: str) -> List[Dict]:
        """Extract relevant sections with multilingual support"""
        relevant_sections = []
        
        for page_content in content['pages']:
            page_num = page_content['page_number']
            
            # Process headings and their associated content
            for heading in page_content['headings']:
                if not self.check_time_limit(f"section analysis page {page_num}"):
                    break
                
                # Find content associated with this heading
                associated_content = self._find_associated_content(
                    heading, page_content['body_text'], content['pages'], page_num
                )
                
                # Calculate relevance with multilingual support
                combined_text = f"{heading['text']} {associated_content}"
                relevance_score = self.calculate_semantic_relevance_multilingual(combined_text, persona, job)
                
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
                
                relevance_score = self.calculate_semantic_relevance_multilingual(text_block['text'], persona, job)
                
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

    def _find_associated_content(self, heading: Dict, body_text: List[Dict], 
                               all_pages: List[Dict], current_page: int) -> str:
        """Find content associated with a heading"""
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

    def _is_heading_advanced(self, text_block: Dict) -> bool:
        """Advanced heading detection with multiple criteria"""
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
        heading_patterns = [
            r'^\d+\.\s+[A-Z]',              # 1. Introduction
            r'^\d+\.\d+\s+[A-Z]',           # 1.1 Overview
            r'^[A-Z][A-Z\s]{3,}$',          # ALL CAPS headings
            r'^Chapter\s+\d+',              # Chapter 1
            r'^Section\s+\d+',              # Section 1
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Keyword-based detection
        heading_keywords = {
            'introduction', 'overview', 'summary', 'conclusion', 'abstract',
            'background', 'methodology', 'results', 'discussion', 'references'
        }
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in heading_keywords):
            return True
        
        return False

    def create_subsection_analysis_advanced(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Advanced subsection analysis with persona-specific insights"""
        subsections = []
        
        for section in sections:
            if not self.check_time_limit("subsection analysis"):
                break
            
            content = section.get('content', '')
            if not content or len(content.strip()) < 20:
                continue
            
            # Advanced content refinement
            refined_text = self._refine_content_for_persona(content, persona, job)
            
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

    def _refine_content_for_persona(self, content: str, persona: str, job: str) -> str:
        """Refine content specifically for the target persona and job"""
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

    def _get_persona_focus_areas(self, persona: str) -> List[str]:
        """Get focus areas for the persona"""
        return self.persona_knowledge.get(persona, {}).get('focus_areas', [])

    def _get_job_alignment(self, job: str) -> str:
        """Get job alignment description"""
        return self.job_patterns.get(job, {}).get('focus', 'General content')

    def _generate_title_selection_reasoning_multilingual(self, section: Dict, persona: str, job: str) -> str:
        """Generate multilingual reasoning for why a title/section was selected"""
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

    def rank_and_filter_results_advanced(self, sections: List[Dict], subsections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Advanced ranking and filtering with improved algorithms"""
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
        """Process documents with multilingual support"""
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
                content = self.extract_document_content_multilingual(pdf_path)
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
                selection_reasoning = self._generate_title_selection_reasoning_multilingual(section, persona, job)
                
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
                selection_reasoning = self._generate_title_selection_reasoning_multilingual(subsection, persona, job)
                
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
        """Create empty result structure"""
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
                "algorithm_version": "2.0_advanced"
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

def main():
    parser = argparse.ArgumentParser(description='Advanced Persona-Driven Document Intelligence v2.0')
    parser.add_argument('--input', default='input', help='Input directory with PDFs')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--persona', required=True, help='Persona (e.g., researcher, student, analyst)')
    parser.add_argument('--job', required=True, help='Job to be done (e.g., literature review, exam preparation)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Adobe Hackathon: Advanced Persona Intelligence v2.0")
    print(f"Persona: {args.persona}")
    print(f"Job: {args.job}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Initialize processor
    processor = AdvancedPersonaIntelligence()
    
    # Process documents
    result = processor.process_documents_multilingual(args.input, args.persona, args.job)
    
    # Save result
    output_file = os.path.join(args.output, 'advanced_persona_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nAdvanced analysis completed successfully!")
    print(f"Results saved to: {output_file}")
    print(f"Found {result['metadata']['total_sections_found']} relevant sections")
    print(f"Found {result['metadata']['total_subsections_found']} relevant subsections")
    print(f"Processing time: {result['metadata']['processing_time_seconds']} seconds")

if __name__ == "__main__":
    main() 