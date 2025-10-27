import cv2
import torch
import numpy as np

from helpers.cnn_model import CNNModel
from helpers.frame_processing import process_frame
from helpers.process_boxes import process_boxes


# load the pip recogniser model
cnn_model = CNNModel()
cnn_model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

# video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to capture and display frames
frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if s is pressed, save the current frame as frame_{timestamp}.png
    key = cv2.waitKey(1)
    if key == ord('s'):
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        cv2.imwrite(f'frame_{timestamp}.png', frame)
        print(f'Saved frame_{timestamp}.png')

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # process the frame to get bounding boxes every 5 frames
    if frame_count % 5 == 0:
        #print("Processing frame:", frame_count)
        bounding_boxes = process_frame(frame)

    # copy the frame to draw on
    img_contours = frame.copy()

    # process the bounding boxes to get the sum of the labels and draw the correct bounding boxes
    label_sum = process_boxes(cnn_model, frame, bounding_boxes, img_contours)

    # put a rectangle in the top left corner, with a text in it showing the sum of the labels as \sigma = {label_sum}
    cv2.rectangle(img_contours, (10, 10), (230, 60), (0, 0, 0), -1)
    cv2.putText(img_contours, f'Sum: {label_sum}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # display the resulting frame with contours
    cv2.imshow('Camera Feed', img_contours)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    frame_count += 1

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
