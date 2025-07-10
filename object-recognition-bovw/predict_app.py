import streamlit as st
import cv2
import numpy as np
import joblib
from features import extract_sift_features
import tempfile
import os

# Load trained models
NUM_CLUSTERS = 150
kmeans = joblib.load("model/dictionary.pkl")
clf = joblib.load("model/svm_model.pkl")
label_names = joblib.load("model/label_names.pkl")

def predict_image_array(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    desc = cv2.SIFT_create().detectAndCompute(gray, None)[1]
    if desc is None:
        return "No features detected", 0.0
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(NUM_CLUSTERS + 1))
    hist = hist.reshape(1, -1)
    prediction = clf.predict(hist)[0]
    proba = clf.predict_proba(hist)[0]
    return label_names[prediction], proba[prediction]

def main():
    st.title("🧠 Object Recognition with BoVW + SVM")
    st.write("Upload an image (e.g., dress, bag, footwear) to classify.")

    uploaded_file = st.file_uploader("📤 Upload test image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image from upload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", caption="Uploaded Image")

        # Predict
        label, confidence = predict_image_array(image)
        st.success(f"✅ Predicted: **{label}** with confidence **{confidence*100:.2f}%**")

if __name__ == "__main__":
    main()
