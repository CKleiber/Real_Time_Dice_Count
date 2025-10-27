import cv2
from torchvision import transforms


# transformations to apply to the die face images to prepare them for the CNN
transform = transforms.Compose([
    # make tensor and therefore scale to [0, 1]
    transforms.ToTensor(),
    # convert to grayscale
    transforms.Grayscale(num_output_channels=1),
    # normalize to [-1, 1]
    transforms.Normalize((0.5,), (0.5,))
])

# take the frame and the bounding box and convert to a usable input for the CNN
def image2input(frame, x, y, w, h):
    die_face = frame[y:y+w, x:x+h]
    gray_die_face = cv2.cvtColor(die_face, cv2.COLOR_BGR2GRAY)
    resized_die_face = cv2.resize(gray_die_face, (28, 28))

    _, binary_die_face = cv2.threshold(resized_die_face, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
    # invert
    binary_die_face = cv2.bitwise_not(binary_die_face)

    # transform and return
    return transform(binary_die_face)
