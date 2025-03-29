from setuptools import setup, find_packages

setup(
    name="wise_nutrition",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "fastapi",
        "uvicorn",
        "weaviate-client",
        "openai",
        "pypdf",
        "python-multipart",
        "tiktoken",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            "pytest-asyncio",
        ],
        "prod": [
            "uvloop",
            "httptools",
            "gunicorn",
        ]
    },
    description="A RAG-based nutrition advisor using LangChain, FastAPI, and Weaviate",
    author="Andre Machon",
    author_email="ajbmachon2n@gmail.com",
    python_requires=">=3.11",
) 