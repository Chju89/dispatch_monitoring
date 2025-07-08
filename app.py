import streamlit as st
import os
from PIL import Image

st.title("Dispatch Monitoring App")

# Dummy display
image_folder = "data/raw/frames"
if os.path.exists(image_folder):
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    for file in files[:5]:  # show 5 images
        st.image(os.path.join(image_folder, file), caption=file)
        st.selectbox("Label", ["dish/empty", "dish/not_empty", "dish/kakigori",
                               "tray/empty", "tray/not_empty", "tray/kakigori"], key=file)
        st.button("Submit Feedback", key="btn_"+file)
else:
    st.warning("No images found in data/raw/frames/")
