#!/bin/bash
# setup.sh - One-time setup script with caching

echo "========================================="
echo "Executive Order Analysis - One-Time Setup"
echo "========================================="
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies (using cache)
echo "Installing dependencies (using pip cache)..."
pip install --cache-dir=.pip-cache -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/{raw,processed,embeddings}
mkdir -p output/plots
mkdir -p .cache
mkdir -p models
mkdir -p .pip-cache

# Download models once
echo ""
echo "========================================="
echo "Downloading NLP models (one-time only)"
echo "========================================="

# Check if models are already downloaded
if [ -f ".cache/.download_complete" ]; then
    echo "Models already downloaded. Skipping download."
    cat .cache/.download_complete
    echo ""
    
    read -p "Download again? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/download_models.py
    fi
else
    echo "Downloading models for first time..."
    python scripts/download_models.py
fi

# Create .env file for local paths
echo ""
echo "Creating .env file with local paths..."
cat > .env << EOF
# Local paths for caching
TRANSFORMERS_CACHE=$(pwd)/models/transformers
HF_HOME=$(pwd)/models/huggingface
TORCH_HOME=$(pwd)/models/torch
NLTK_DATA=$(pwd)/.cache/nltk_data
EOF

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the analysis: python scripts/run_analysis.py"
echo "3. Or use the Makefile: make run"
echo ""
echo "All models are cached locally in:"
echo "  - ./models/ (transformer models)"
echo "  - ./.cache/ (NLTK data)"
echo "  - ./.pip-cache/ (pip packages)"
echo ""