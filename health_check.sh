#!/bin/bash
# Health check script for Crop Disease AI

echo "🌿 Crop Disease AI — Health Check"
echo "=================================="

# Check API
echo "\n1. Checking API..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$response" = "200" ]; then
    echo "   API: ✓ Running"
else
    echo "   API: ✗ Not running (start with: uvicorn src.api:app --port 8000)"
fi

# Check Streamlit
echo "\n2. Checking Streamlit..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
if [ "$response" = "200" ]; then
    echo "   Streamlit: ✓ Running"
else
    echo "   Streamlit: ✗ Not running (start with: streamlit run app.py)"
fi

# Check model files
echo "\n3. Checking model files..."
models=("models/resnet50_balanced.pth" "models/fusion_mlp_best.pth" 
        "models/xgboost_tuned.pkl" "models/tabular_scaler_v2.pkl")
for model in "${models[@]}"; do
    if [ -f "$model" ]; then
        echo "   $model: ✓"
    else
        echo "   $model: ✗ MISSING"
    fi
done

echo "\n=================================="
echo "Health check complete!"
