import numpy as np
import cv2

def merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array):
    """
    This function takes three 2D arrays representing the red, green, and blue channels of an image,
    combines them into a single RGB image, and returns the saved image.
    Parameters:
    - Red_2D_array: 2D array of the red channel (values between 0 and 255)
    - Green_2D_array: 2D array of the green channel (values between 0 and 255)
    - Blue_2D_array: 2D array of the blue channel (values between 0 and 255)
    Returns:
    - bgr_image: The combined image in BGR format (as used by OpenCV)
    """
    # Ensure input arrays are uint8
    Red_2D_array = np.asarray(Red_2D_array, dtype=np.uint8)
    Green_2D_array = np.asarray(Green_2D_array, dtype=np.uint8)
    Blue_2D_array = np.asarray(Blue_2D_array, dtype=np.uint8)

    # Pre-allocate and assign channels directly (faster than np.dstack)
    bgr_image = np.empty((*Blue_2D_array.shape, 3), dtype=np.uint8)
    bgr_image[:, :, 0] = Blue_2D_array
    bgr_image[:, :, 1] = Green_2D_array
    bgr_image[:, :, 2] = Red_2D_array
    
    return bgr_image