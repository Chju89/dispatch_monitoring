# app/classifier.py
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class ObjectClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        self.label_map = {
            0: 'dish_empty',
            1: 'dish_kakigori',
            2: 'dish_not_empty',
            3: 'tray_empty',
            4: 'tray_kakigori',
            5: 'tray_not_empty'
        }

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        try:
            if image is None or image.shape[0] < 10 or image.shape[1] < 10:
                return "invalid"

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                pred = torch.argmax(out, dim=1).item()
                return self.label_map.get(pred, "unknown")
        except Exception as e:
            return "error"
