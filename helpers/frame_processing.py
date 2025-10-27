import cv2
from networkx import adjacency_matrix
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# watershed segmentation to separate connected die faces
def watershed_segmentation(threshed_image):
    # compute the distance transform
    distance_transform = cv2.distanceTransform(threshed_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # unknown region
    unknown = cv2.subtract(threshed_image, sure_fg)

    # markers
    _, markers = cv2.connectedComponents(sure_fg)

    # add one to all the labels so that sure regions are marked with positive integers
    markers = markers + 1

    # mark the unknown region with zero
    markers[unknown == 255] = 0

    # apply watershed
    markers = cv2.watershed(cv2.cvtColor(threshed_image, cv2.COLOR_GRAY2BGR), markers)

    # extract bounding boxes from the markers
    bounding_boxes = []
    for marker in range(2, markers.max() + 1):
        mask = np.uint8(markers == marker) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

# the aim of this function is to use blob detection to find the die faces in the frame
def process_frame(frame):
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0.5)
    
    # thresholding using Otsu's method
    _, threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel, iterations=1)

    
    # detect blobs using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 75
    params.maxArea = 300

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(threshed)

    # invert image and detect again
    threshed_inv = cv2.bitwise_not(threshed)
    keypoints_inv = detector.detect(threshed_inv)

    # plot all the keypoints found
    all_keypoints = keypoints + keypoints_inv

    # get the pixels of each key point and turn it white in the threshed image to fill up the dice faces
    for keypoint in all_keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        radius = int(keypoint.size / 2) + 3  # add 3 pixels to make sure the dot is fully covered

        cv2.circle(threshed, (x, y), radius, (255, 255, 255), -1)

    # closing transformation to fill holes
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel, iterations=2)

    # watershed algorithm to separate connected die faces
    bounding_boxes = watershed_segmentation(threshed)
    final_bounding_boxes = []

    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        
        # make sure the box has an aspect ratio between 0.7 and 1.3
        aspect_ratio = w / h
        if 0.7 <= aspect_ratio <= 1.3:
            # also check that it has a minimum size of 0.5 % of the frame area and a maximum size of 6.5 % of the frame area
            if 0.005 * frame.shape[0] * frame.shape[1] <  w * h <= 0.065 * frame.shape[0] * frame.shape[1]:
                size = (h+w)//2

                if x + size >= frame.shape[1]:
                    size = frame.shape[1] - x - 1

                if y + size >= frame.shape[0]:
                    size = frame.shape[0] - y - 1

                final_bounding_boxes.append((x, y, size, size))
            

    return final_bounding_boxes
    ##########################################
    # plot the key points and the connections

    plt.imshow(gray, cmap='gray')
    #plt.scatter(*zip(*key_points_coords), color='red')

    #for i in range(len(key_points_coords)):
    #    for j in range(len(key_points_coords)):
    #        if adjacency_matrix[i][j]:
    #            x_values = [key_points_coords[i][0], key_points_coords[j][0]]
    #            y_values = [key_points_coords[i][1], key_points_coords[j][1]]
    #            plt.plot(x_values, y_values, color='blue')

    #for cluster_label in set(labels):
    #    cluster_points = [key_points_coords[i] for i in range(len(key_points_coords)) if labels[i] == cluster_label]
    #    
    #    # connect the cluster points with blue lines
    #    for i in range(len(cluster_points)):
    #        for j in range(i + 1, len(cluster_points)):
    #            x_values = [cluster_points[i][0], cluster_points[j][0]]
    #            y_values = [cluster_points[i][1], cluster_points[j][1]]
    #            plt.plot(x_values, y_values, color='blue')

    # draw bounding boxes in green
    for (x, y, w, h) in final_bounding_boxes:
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()


# load frame \video_frames\1\frame_1761483238.png
#frame = cv2.imread('video_frames/6/frame_1761483850.png')
#process_frame(frame)