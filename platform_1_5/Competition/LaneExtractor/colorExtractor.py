import cv2
import numpy as np

def yellow_color_extractor(image):
    """
    This function takes an image and extracts the yellow color regions.
    Parameters:
    - image: Input image in BGR format
    Returns:
    - yellow_mask: A binary mask where yellow regions are white and the rest are black
    """

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where yellow colors are white and the rest are black
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    return yellow_mask

def white_color_extractor(image):
    """
    This function takes an image and extracts the white color regions.
    Parameters:
    - image: Input image in BGR format
    Returns:
    - white_mask: A binary mask where white regions are white and the rest are black
    """

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Create a binary mask where white colors are white and the rest are black
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    return white_mask

def black_color_extractor(image):
    """
    This function takes an image and extracts the black color regions.
    Parameters:
    - image: Input image in BGR format
    Returns:
    - black_mask: A binary mask where black regions are white and the rest are black
    """

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Create a binary mask where black colors are white and the rest are black
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

    return black_mask