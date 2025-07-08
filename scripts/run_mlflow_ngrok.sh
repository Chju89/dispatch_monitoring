#!/bin/bash

echo "🔁 Starting MLflow server at http://localhost:5000 ..."
mlflow ui --host 0.0.0.0 --port 5000 &

sleep 3

echo "🌐 Starting ngrok to expose MLflow server ..."
ngrok http 5000

