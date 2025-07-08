# Classification model logic
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path

model = torch.load("models/classification/resnet18_dispatch.pt", map_location="cpu")
model.eval()

crop_dir = Path("data/processed/tracking/crops")
output_csv = Path("data/processed/tracking/classification_result.csv")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

results = []
for cls_folder in crop_dir.iterdir():
    for track_folder in cls_folder.iterdir():
        for img_path in track_folder.glob("*.jpg"):
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            results.append({
                "filename": img_path.name,
                "true_class": cls_folder.name,
                "track_id": track_folder.name,
                "predicted": pred
            })

pd.DataFrame(results).to_csv(output_csv, index=False)
print("✅ Classification done. Saved to:", output_csv)

