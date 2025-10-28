"""
Setup configuration for the HooYouKnow interview extraction pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file, "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="hooyouknow",
    version="1.0.0",
    author="HooYouKnow Team",
    author_email="contact@hooyouknow.com",
    description="Automated pipeline for converting interview audio into structured newsletter content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/interview-extractor",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/interview-extractor/issues",
        "Documentation": "https://github.com/yourusername/interview-extractor#readme",
        "Source Code": "https://github.com/yourusername/interview-extractor",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hooyouknow=hooyouknow.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hooyouknow": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "interview",
        "transcription",
        "newsletter",
        "audio",
        "whisper",
        "claude",
        "gpt-4",
        "content-extraction",
        "automation",
    ],
)
