import cv2

class DenseDetector:
    def __init__(self, step_size=20):
        self.step_size = step_size

    def detect(self, img):
        keypoints = []
        for y in range(0, img.shape[0], self.step_size):
            for x in range(0, img.shape[1], self.step_size):
                kp = cv2.KeyPoint(x, y, self.step_size)
                keypoints.append(kp)
        return keypoints


def detect_sift(img_gray):
    try:
        sift = cv2.SIFT_create()
        return sift.detect(img_gray, None)
    except:
        return []
