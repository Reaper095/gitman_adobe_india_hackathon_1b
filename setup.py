#!/usr/bin/env python3
"""
Setup script for Advanced Persona-Driven Document Intelligence v3.0
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="persona-intelligence",
    version="3.0.0",
    author="Adobe Hackathon Team",
    author_email="hackathon@adobe.com",
    description="Advanced Persona-Driven Document Intelligence System",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adobe-hackathon/persona-intelligence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docker": [
            "docker>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "persona-intelligence=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="nlp, document-analysis, persona, multilingual, pdf-processing, ai",
    project_urls={
        "Bug Reports": "https://github.com/adobe-hackathon/persona-intelligence/issues",
        "Source": "https://github.com/adobe-hackathon/persona-intelligence",
        "Documentation": "https://github.com/adobe-hackathon/persona-intelligence#readme",
    },
) 