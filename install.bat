@echo off
REM Advanced Persona-Driven Document Intelligence v3.0 Installation Script
REM "Connect What Matters — For the User Who Matters"

echo 🚀 Installing Advanced Persona-Driven Document Intelligence v3.0
echo ================================================================

REM Check if Python 3.9+ is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Download spaCy model
echo 🤖 Downloading spaCy English model...
python -m spacy download en_core_web_sm

REM Download NLTK data
echo 📖 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

REM Create directories
echo 📁 Creating directories...
if not exist input mkdir input
if not exist output mkdir output

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Place PDF files in the 'input' directory
echo 3. Run the analysis: python main.py --persona researcher --job literature_review
echo.
echo 📖 For more information, see README.md
echo.
echo 🔗 Quick start examples:
echo   python main.py --persona student --job exam_preparation
echo   python main.py --persona analyst --job data_analysis
echo   python main.py --persona manager --job decision_making
echo.
echo 💡 Tip: Use 'python main.py --help' to see all available options
echo.
echo ✨ "Connect What Matters — For the User Who Matters" ✨
pause 