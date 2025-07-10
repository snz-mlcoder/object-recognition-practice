import streamlit as st
import cv2
import numpy as np
from utils import DenseDetector, detect_sift
from PIL import Image
import io
import os

def save_image_to_disk(img, filename):
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", filename)
    cv2.imwrite(path, img)

def main():
    st.title("🔍 Dense vs SIFT Feature Detector")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and prepare image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        st.subheader("🖼️ Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

        # --- Dense Detector ---
        st.subheader("🔶 Dense Feature Detector")
        step = st.slider("🔧 Dense Step Size (lower = more points)", min_value=5, max_value=60, value=20, step=5)

        dense = DenseDetector(step_size=step)
        dense_kp = dense.detect(img_gray)
        img_dense = cv2.drawKeypoints(img.copy(), dense_kp, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        st.image(cv2.cvtColor(img_dense, cv2.COLOR_BGR2RGB), channels="RGB")
        st.caption(f"{len(dense_kp)} dense keypoints detected.")

        # Save to disk
        save_image_to_disk(img_dense, "dense_output.png")

        # Download button
        dense_pil = Image.fromarray(cv2.cvtColor(img_dense, cv2.COLOR_BGR2RGB))
        buf_dense = io.BytesIO()
        dense_pil.save(buf_dense, format="PNG")
        st.download_button(
            label=" Download Dense Output",
            data=buf_dense.getvalue(),
            file_name="dense_output.png",
            mime="image/png"
        )

        # --- SIFT Detector ---
        st.subheader("🔷 SIFT Feature Detector")
        sift_kp = detect_sift(img_gray)
        img_sift = cv2.drawKeypoints(img.copy(), sift_kp, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        st.image(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB), channels="RGB")
        st.caption(f"{len(sift_kp)} SIFT keypoints detected.")

        # Save to disk
        save_image_to_disk(img_sift, "sift_output.png")

        # Download button
        sift_pil = Image.fromarray(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
        buf_sift = io.BytesIO()
        sift_pil.save(buf_sift, format="PNG")
        st.download_button(
            label=" Download SIFT Output",
            data=buf_sift.getvalue(),
            file_name="sift_output.png",
            mime="image/png"
        )


if __name__ == "__main__":
    main()
