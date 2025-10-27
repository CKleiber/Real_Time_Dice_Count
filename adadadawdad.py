import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# TODO: REDO THIS FUNCTION

# Function to process a frame to identify the bounding boxes of the die faces
def process_frame(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0.5)
    # thresholding using Otsu's method
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(threshed == 255) > np.sum(threshed == 0):
        threshed = cv2.bitwise_not(threshed)

    # morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=2)

    white_pixels = np.column_stack(np.where(threshed == 255))
    
    # if not enough white pixels, return empty list
    if len(white_pixels) < 50:
        return []

    # evaluate the clusters by using calinski_harabasz_score
    scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(white_pixels)
        labels = kmeans.labels_
        score = calinski_harabasz_score(white_pixels, labels)
        scores.append(score)

    optimal_k = np.argmax(scores) + 2

    # clusterin with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(white_pixels)
    labels = kmeans.labels_

    # calculate bounding boxes for each cluster
    bounding_boxes = []
    for i in range(optimal_k):
        cluster_points = white_pixels[labels == i]
        x, y, w, h = cv2.boundingRect(cluster_points)

        # check whether the aspect ratio is acceptable between 0.8 and 1.2
        aspect_ratio = w / h
        if 0.8 <= aspect_ratio <= 1.2:
            size = (w+h)//2
            bounding_boxes.append((x, y, size, size))

    # for each bounding box, calculate the order
    sorting_number = []
    for (x, y, w, h) in bounding_boxes:
        sorting_number.append(x * frame.shape[1] + y)

    # sort bounding boxes according to the sorting number
    bounding_boxes = [box for _, box in sorted(zip(sorting_number, bounding_boxes))]

    return bounding_boxes
