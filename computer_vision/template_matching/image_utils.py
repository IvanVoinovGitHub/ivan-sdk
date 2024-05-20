import numpy as np
from scipy.ndimage import zoom

def downsample_image_scipy(image_array, scale_factor):
    """
    Downsamples an image using Scipy.

    Parameters:
    image_array (numpy.ndarray): The original image as a NumPy array.
    scale_factor (float): The factor by which to downsample the image.

    Returns:
    numpy.ndarray: The downsampled image.
    """
    # Calculate the zoom factor
    zoom_factor = 1 / scale_factor
    # Downsample the image
    downsampled_image = zoom(image_array, zoom_factor, order=3)  # order=3 for cubic interpolation
    return downsampled_image


def pad_or_crop_image(image, target_shape):
    """
    Pad or crop an image to the target shape.

    Parameters:
    - image (numpy.ndarray): The input image.
    - target_shape (tuple): The target shape (height, width) for the image.

    Returns:
    - padded_image (numpy.ndarray): The padded or cropped image.
    """
    current_shape = image.shape
    
    # Initialize the padded_image with white pixels (255 for an 8-bit grayscale image)
    padded_image = np.ones(target_shape, dtype=image.dtype) * 255
    
    offset_x = (target_shape[1] - current_shape[1]) // 2
    offset_y = (target_shape[0] - current_shape[0]) // 2
    
    # Determine the region of the padded_image to replace with the original image
    y1 = max(0, offset_y)
    y2 = min(target_shape[0], offset_y + current_shape[0])
    x1 = max(0, offset_x)
    x2 = min(target_shape[1], offset_x + current_shape[1])
    
    # Determine the region of the original image to place in the padded_image
    img_y1 = max(0, -offset_y)
    img_y2 = img_y1 + (y2 - y1)
    img_x1 = max(0, -offset_x)
    img_x2 = img_x1 + (x2 - x1)
    
    # Place the original image into the padded_image
    padded_image[y1:y2, x1:x2] = image[img_y1:img_y2, img_x1:img_x2]
    
    return padded_image
