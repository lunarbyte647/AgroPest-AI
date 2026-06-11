import cv2
import numpy as np

def analyze_health(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = np.count_nonzero(green_mask)

    total_pixels = image.shape[0] * image.shape[1]

    green_percentage = (green_pixels / total_pixels) * 100

    damage_percentage = 100 - green_percentage

    return round(green_percentage, 2), round(damage_percentage, 2)