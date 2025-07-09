#!/bin/bash

# Start MLflow UI in background
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 &

# Run Streamlit UI
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
