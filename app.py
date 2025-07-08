import streamlit as st
import cv2
from pathlib import Path
import pandas as pd

st.title("📦 Dispatch Monitoring Feedback UI")

frame_dir = Path("data/processed/tracking/frames_with_id")
csv_path = Path("data/processed/tracking/classification_result.csv")
feedback_path = Path("data/processed/feedback.csv")

if not frame_dir.exists():
    st.error("❌ Không tìm thấy frame đã tracking")
    st.stop()
if not csv_path.exists():
    st.error("❌ Không tìm thấy file classification_result.csv")
    st.stop()

df = pd.read_csv(csv_path)
frames = sorted(list(frame_dir.glob("*.jpg")))

if "feedback" not in st.session_state:
    st.session_state.feedback = []

idx = st.slider("Chọn frame", 0, len(frames) - 1, 0)
frame_path = frames[idx]
frame = cv2.imread(str(frame_path))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
st.image(frame, caption=frame_path.name, use_column_width=True)

track_ids = df["track_id"].unique()
selected_id = st.selectbox("Chọn Track ID cần phản hồi", track_ids)
predicted_label = df[df["track_id"] == selected_id]["predicted"].values[0]
true_label = st.text_input("Nhãn đúng là:", "")

if st.button("📥 Gửi phản hồi"):
    st.session_state.feedback.append({
        "frame": frame_path.name,
        "track_id": selected_id,
        "predicted": predicted_label,
        "corrected": true_label
    })
    st.success("✅ Đã lưu phản hồi")

if st.button("💾 Xuất file feedback"):
    pd.DataFrame(st.session_state.feedback).to_csv(feedback_path, index=False)
    st.success(f"✅ Feedback saved to {feedback_path}")

