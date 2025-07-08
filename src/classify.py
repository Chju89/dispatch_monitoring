# Classification model logic
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path

model = models.resnet18(pretrained=False, num_classes=6)
state_dict = torch.load("models/classification/resnet18_dispatch.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

id_to_class = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

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
                label_name = id_to_class[pred]
            results.append({
                "filename": img_path.name,
                "true_class": cls_folder.name,
                "track_id": track_folder.name,
                "predicted": pred,
                 "predicted_label": label_name 
            })

pd.DataFrame(results).to_csv(output_csv, index=False)
print("✅ Classification done. Saved to:", output_csv)

