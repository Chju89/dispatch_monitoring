#!/bin/bash

VIDEO_PATH=$1

if [ -z "$VIDEO_PATH" ]; then
  echo "❌ Vui lòng cung cấp đường dẫn video."
  echo "Usage: ./scripts/run_pipeline.sh path/to/video.mp4"
  exit 1
fi

echo "🚀 Bắt đầu tracking..."
python src/tracking.py --video_path "$VIDEO_PATH"
if [ $? -ne 0 ]; then
  echo "❌ Tracking thất bại!"
  exit 1
fi

echo "✅ Tracking hoàn tất."

echo "🔍 Bắt đầu classification..."
python src/classify.py --video_path "$VIDEO_PATH"
if [ $? -ne 0 ]; then
  echo "❌ Classification thất bại!"
  exit 1
fi

echo "✅ Classification hoàn tất. Pipeline kết thúc."

