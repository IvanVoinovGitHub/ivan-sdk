import torch
import numpy as np
from scipy.signal import correlate2d
from PIL import Image


def pillow_cross_correlation(image):
    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)

    # Ensure the image is a 2D array (grayscale)
    if len(image_np.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale)")
    
    # Apply cross-correlation to the image
    correlated_image = correlate2d(image_np, image_np, mode='same', boundary='wrap')

    # Convert the result back to a PIL Image
    correlated_image_pil = Image.fromarray(correlated_image)
    
    return correlated_image_pil


def torch_cross_correlation(image):
    # Convert the image to numpy array
    image_np = image.numpy()
    
    # Apply cross-correlation to the image
    correlated_image = np.zeros_like(image_np)
    for channel in range(image_np.shape[0]):
        correlated_image[channel] = correlate2d(image_np[channel], image_np[channel], mode='same', boundary='wrap')
    
    # Convert the result back to a tensor
    correlated_image_tensor = torch.from_numpy(correlated_image)
    
    return correlated_image_tensor