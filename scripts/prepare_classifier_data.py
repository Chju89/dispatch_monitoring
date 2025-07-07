import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Đường dẫn nguồn và đích
SOURCE_DIR = Path("data/raw/Dataset/Classification")
DEST_DIR = Path("data/processed/classifier")

# Tạo thư mục đích nếu chưa có
DEST_DIR.mkdir(parents=True, exist_ok=True)

total = 0
for object_type in ["dish", "tray"]:
    type_path = SOURCE_DIR / object_type
    if not type_path.exists():
        continue

    for state in os.listdir(type_path):  # empty / not_empty / kakigori
        source_class_path = type_path / state
        if not source_class_path.is_dir():
            continue

        class_name = f"{object_type}_{state}".lower()
        target_class_path = DEST_DIR / class_name
        target_class_path.mkdir(parents=True, exist_ok=True)

        for img_file in tqdm(os.listdir(source_class_path), desc=class_name):
            src_img = source_class_path / img_file
            dst_img = target_class_path / img_file
            shutil.copy2(src_img, dst_img)
            total += 1

print(f"\n✅ Đã gộp tổng cộng {total} ảnh vào data/processed/classifier/ với 6 class.")

