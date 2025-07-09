import pandas as pd
from collections import Counter
from pathlib import Path

# Đường dẫn đến file classification result
input_csv = Path("data/processed/tracking/classification_result.csv")
output_csv = Path("data/processed/tracking/summary.csv")

# Đọc dữ liệu
df = pd.read_csv(input_csv)

# Gộp theo track_id
grouped = df.groupby("track_id")

summary = []
for track_id, group in grouped:
    most_common = Counter(group["predicted_label"]).most_common(1)[0][0]
    true_class = group["true_class"].iloc[0]
    summary.append({
        "track_id": track_id,
        "true_class": true_class,
        "predicted_label": most_common
    })

# Ghi file kết quả
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_csv, index=False)
print(f"[✅] Saved summary.csv with final predicted label per object to: {output_csv}")

