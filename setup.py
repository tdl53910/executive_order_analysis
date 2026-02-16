"""
Setup script for executive-orders-analysis package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="executive-orders-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered analysis of U.S. executive orders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/executive-orders-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eo-analyze=scripts.run_analysis:main",
            "eo-export=scripts.export_results:main",
        ],
    },
)