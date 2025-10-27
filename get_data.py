import cv2
import os
import tqdm
import numpy as np

from helpers.frame_processing import process_frame
from helpers.image2input import image2input

"""
Script to extract die face images from video frames and save them in the data/ folder for training the CNN model.

The images all contain only one side of the die, i.e. only 1s, only 2s, ..., only 6s.

The video frames are stored in video_frames/{pips}/ where {pips} is the number of pips on the die faces (1-6).
The extracted die face images are saved in data/{pips}/ with filenames die_{i}_{f}, 
where {i} is the index of the die face in the frame and {f} is the original filename.


After the data has been generated, inspect the images and remove any faulty images manually.
"""


# prepare processing progress bar
pbar = tqdm.tqdm(total=range(1, 7).__len__())

# iterate over all pips
for pips in range(1, 7):
    # read all the files in video_frames/{pips}/
    files = os.listdir(f'video_frames/{pips}/')
    # for each file, extract the die faces and save them in data/{pips}/
    for f in files:
        filename = f'video_frames/{pips}/{f}'
        frame = cv2.imread(filename)

        # get bounding boxes of die faces in the frame
        bounding_boxes = process_frame(frame)

        # cut the image for each detected bounding box and process to prepare for CNN input 
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            die_face = image2input(frame, x, y, w, h)

            # die face is now a tensor, withthe pixel ranges normalized to [-1, 1].
            # transform to numpy array with pixel values in [0, 255] for saving as image
            die_face = die_face.squeeze().numpy()  # remove channel dimension
            die_face = (die_face * 255).astype(np.uint8)  # scale back to [0, 255]

            # save the image in data/{pips} with filename die_{i}_{f}
            cv2.imwrite(f'data/{pips}/die_{i}_{f}', die_face)

    pbar.update(1)