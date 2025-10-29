#!/bin/bash
echo "🚀 Starting Institutional Football Predictor Pro..."
echo "📊 Loading professional prediction engine..."
echo "📦 Installing dependencies if needed..."
pip install streamlit numpy pandas plotly scipy --quiet
echo "⚽ App will open at http://localhost:8501"
streamlit run streamlit_app.py --server.port 8501
