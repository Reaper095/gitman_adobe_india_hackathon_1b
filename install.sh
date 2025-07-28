#!/bin/bash

# Advanced Persona-Driven Document Intelligence v3.0 Installation Script
# "Connect What Matters — For the User Who Matters"

echo "🚀 Installing Advanced Persona-Driven Document Intelligence v3.0"
echo "================================================================"

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python 3.9 or higher is required. Current version: $python_version"
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🤖 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "📖 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create directories
echo "📁 Creating directories..."
mkdir -p input output

# Make main.py executable
chmod +x main.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Place PDF files in the 'input' directory"
echo "3. Run the analysis: python main.py --persona researcher --job literature_review"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "🔗 Quick start examples:"
echo "  python main.py --persona student --job exam_preparation"
echo "  python main.py --persona analyst --job data_analysis"
echo "  python main.py --persona manager --job decision_making"
echo ""
echo "💡 Tip: Use 'python main.py --help' to see all available options"
echo ""
echo "✨ 'Connect What Matters — For the User Who Matters' ✨" 