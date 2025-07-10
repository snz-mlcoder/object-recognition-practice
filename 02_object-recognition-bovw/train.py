import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from tqdm import tqdm
from features import load_all_descriptors
import os

# Parameters
NUM_CLUSTERS = 150
DATA_DIR = "Images"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Step 1: Load descriptors
print("📦 Extracting features from all images...")
des_list, img_paths, labels, label_names = load_all_descriptors(DATA_DIR)

# Step 2: Build visual dictionary (KMeans)
print("📊 Building visual dictionary...")
all_descriptors = np.vstack(des_list)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(all_descriptors)
joblib.dump(kmeans, os.path.join(MODEL_DIR, "dictionary.pkl"))

# Step 3: Build histograms
def compute_histogram(descriptors, kmeans):
    visual_words = kmeans.predict(descriptors)
    histogram, _ = np.histogram(visual_words, bins=np.arange(NUM_CLUSTERS + 1))
    return histogram

X = []
for descriptors in tqdm(des_list):
    hist = compute_histogram(descriptors, kmeans)
    X.append(hist)
X = np.array(X)
y = np.array(labels)

# Step 4: Train SVM classifier
print("🧠 Training SVM...")
clf = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
clf.fit(X, y)
joblib.dump(clf, os.path.join(MODEL_DIR, "svm_model.pkl"))
joblib.dump(label_names, os.path.join(MODEL_DIR, "label_names.pkl"))

print("✅ Training complete!")
