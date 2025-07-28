#!/usr/bin/env python3
"""
Enhanced Adobe Hackathon: Advanced Persona-Driven Document Intelligence v3.0
"Connect What Matters — For the User Who Matters"

Main entry point for the modular document analysis system.
"""

import json
import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from src.persona_processor import AdvancedPersonaIntelligence


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Advanced Persona-Driven Document Intelligence v3.0')
    parser.add_argument('--input', default='input', help='Input directory with PDFs')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--persona', required=True, help='Persona (e.g., researcher, student, analyst)')
    parser.add_argument('--job', required=True, help='Job to be done (e.g., literature review, exam preparation)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Adobe Hackathon: Advanced Persona Intelligence v3.0")
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