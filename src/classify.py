import cv2
import torch
import pickle
import argparse
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# CLI arguments
parser = argparse.ArgumentParser(description="Run classification on tracked objects in video.")
parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
args = parser.parse_args()
video_path = args.video_path

# Paths
bbox_log_path = Path("data/processed/tracking/logs/bbox_tracking_log.pkl")
output_csv = Path("data/processed/tracking/logs/classification_result.csv")
ui_objects_path = Path("data/processed/tracking/logs/ui_objects.pkl")

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False, num_classes=6)
state_dict = torch.load("models/classification/resnet18_dispatch.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Mapping class ID to label
id_to_class = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

# Load bbox log
with open(bbox_log_path, "rb") as f:
    bbox_data = pickle.load(f)

# Open video once
cap = cv2.VideoCapture(video_path)

results = []
ui_data = {}

for frame_idx, objs in bbox_data.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        print(f"[!] Cannot read frame {frame_idx}")
        continue

    height, width = frame.shape[:2]
    ui_data[frame_idx] = []
    for obj in objs:
        cls = obj["object"]
        track_id = obj["id"]
        x, y, w, h = obj["bbox"]

        # Clip bbox to frame size
        x = max(0, x)
        y = max(0, y)
        x2 = min(x + w, width)
        y2 = min(y + h, height)

        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred].item()
            label_name = id_to_class[pred]

        ui_data[frame_idx].append({
            "id": track_id,
            "object": cls,
            "status": label_name.split("_")[1],
            "bbox": obj["bbox"],
            "confidence": round(confidence, 4)
        })

        results.append({
            "frame": frame_idx,
            "track_id": track_id,
            "object": cls,
            "predicted": pred,
            "predicted_label": label_name,
            "confidence": round(confidence, 4)
        })

cap.release()

# Save results
pd.DataFrame(results).to_csv(output_csv, index=False)
with open(ui_objects_path, "wb") as f:
    pickle.dump(ui_data, f)

print("\u2705 Classification done. Logs saved to:", output_csv, "and", ui_objects_path)
