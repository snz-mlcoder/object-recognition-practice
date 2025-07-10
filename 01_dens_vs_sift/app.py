import streamlit as st
import cv2
import numpy as np
from utils import DenseDetector, detect_sift


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

        # --- SIFT Detector ---
        st.subheader("🔷 SIFT Feature Detector")
        sift_kp = detect_sift(img_gray)
        img_sift = cv2.drawKeypoints(img.copy(), sift_kp, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        st.image(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB), channels="RGB")
        st.caption(f"{len(sift_kp)} SIFT keypoints detected.")


if __name__ == "__main__":
    main()
