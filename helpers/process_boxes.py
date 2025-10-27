import cv2
import numpy as np
import torchvision
import torch

from helpers.image2input import image2input


# function to process bounding boxes, i.e.:
# - cut the image within the bounding box
# - prepare it for the CNN model
# - predict the label
# - evalueate the prediction confidence
# - sum up the predicted labels
# - draw the bounding box and the predicted label on the image

def process_boxes(cnn_model, frame, bounding_boxes, img_contours):
    # running sum of the labels
    label_sum = 0

    if len(bounding_boxes) == 0:
        return label_sum

    # get the input tensors for all bounding boxes
    input_tensors = []
    for (x, y, w, h) in bounding_boxes:
        # prepare the die face image for the CNN
        input_tensors.append(image2input(frame, x, y, w, h))

    # batch process all die face images
    outputs = cnn_model(torch.stack(input_tensors))

    # process each output
    predicted_labels = []
    die_bounding_boxes = []
    
    for i, output in enumerate(outputs):
        output = torch.softmax(output.unsqueeze(0), dim=1)

        predicted_label = int(torch.argmax(output)) + 1  # add 1 to match the conversion from index to label

        # add to sum and plot only if the predicted label percentage is above 50%
        if output[0][predicted_label - 1].item() > 0.16:
            # add to total sum
            label_sum += predicted_label
            predicted_labels.append(predicted_label)
            die_bounding_boxes.append(bounding_boxes[i])

    # sort the bounding boxed according to their position in the frame (top to bottom, left to right) and apply the same order to the predicted labels
    sorting_numbers = []
    for (x, y, w, h) in die_bounding_boxes:
        sorting_numbers.append(x * frame.shape[1] + y)
    sorted_indices = np.argsort(sorting_numbers)
    die_bounding_boxes = [die_bounding_boxes[i] for i in sorted_indices]
    predicted_labels = [predicted_labels[i] for i in sorted_indices]

    # plotting
    number_of_dice = len(predicted_labels)
    for i, (predicted_label, (x, y, w, h)) in enumerate(zip(predicted_labels, die_bounding_boxes)):
        
        # plot the bounding boxes with unique colours

        # get colour:
        hue = int(i * 179 / number_of_dice) if number_of_dice > 1 else 0
        hsv_pixel = np.uint8([[[hue, 200, 255]]])  # (H, S, V)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        colour = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

        # plot the box
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), colour, 2)

        # put a handle in the same color on the top left corner of the box with the predicted label
        # filled rectangle
        cv2.rectangle(img_contours, (x, y - 30), (x + 40, y), colour, -1)
        # text om the rectangle
        cv2.putText(img_contours, str(predicted_label), (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    return label_sum
