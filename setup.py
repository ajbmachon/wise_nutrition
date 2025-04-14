"""
Setup script for Wise Nutrition.
"""
from setuptools import setup, find_packages

setup(
    name="wise-nutrition",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "click>=8.0.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-chroma>=0.0.5",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "firebase-admin>=6.0.0",
        "email-validator>=2.0.0",  # For Pydantic's EmailStr validation
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            "pytest-asyncio",
            "chromadb",
            "langchain-community",
        ],
        "prod": [
            "uvloop",
            "httptools",
            "gunicorn",
        ],
        "ingestion": [
            "langchain-unstructured",
            "pdfplumber",
        ]
    },
    entry_points={
        "console_scripts": [
            "nutrition-cli=wise_nutrition.cli.main:cli",
        ],
    },
    description="A RAG-based nutrition advisor using LangChain, FastAPI, and Weaviate",
    author="Andre Machon",
    author_email="ajbmachon2n@gmail.com",
    python_requires=">=3.11",
) 