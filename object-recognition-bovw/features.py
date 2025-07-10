import cv2
import numpy as np
import os

def extract_sift_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def load_all_descriptors(image_dir):
    descriptors_list = []
    image_paths = []
    labels = []
    label_names = sorted(os.listdir(image_dir))
    for label, folder in enumerate(label_names):
        folder_path = os.path.join(image_dir, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            desc = extract_sift_features(img_path)
            if desc is not None:
                descriptors_list.append(desc)
                image_paths.append(img_path)
                labels.append(label)
    return descriptors_list, image_paths, labels, label_names
