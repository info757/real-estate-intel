#!/bin/bash

# Real Estate Intelligence Platform - Prototype Launcher
# This script starts the Streamlit prototype

echo "🏘️  Real Estate Intelligence Platform"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your API keys."
    echo ""
fi

# Check if Qdrant is running (optional check)
echo "🔍 Checking if Qdrant is running..."
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "✅ Qdrant is running"
else
    echo "⚠️  Qdrant is not running. AI features will be limited."
    echo "   To start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
    echo ""
fi

# Start Streamlit
echo "🚀 Launching Streamlit prototype..."
echo ""
streamlit run prototype/app.py

