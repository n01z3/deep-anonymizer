import os
import cv2


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError('Failed to load image at "{}"'.format(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)
    return path
