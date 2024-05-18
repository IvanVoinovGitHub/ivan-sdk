import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import datasets
from scipy.ndimage import rotate
import plotly.graph_objs as go
import plotly.io as pio


def scipy_example():
    """
    Generates example background image and filter from scipy built-in dataset

    Parameters:
    None

    Returns:
    None
    """
    rng = np.random.default_rng()
    background_image = datasets.face(gray=True) - datasets.face(gray=True).mean()
    filter = np.copy(background_image[300:365, 670:750])  # right eye
    filter -= filter.mean()
    background_image = background_image + rng.standard_normal(background_image.shape) * 50  # add noise
    return background_image, filter


def cross_correlation(background_image, filter, return_max_loc=False):
    """
    Compute the cross-correlation between a background image and a filter image.

    Parameters:
    - background_image (numpy.ndarray): The background image.
    - filter (numpy.ndarray): The filter image.
    - return_max_loc (bool, optional): Whether to return the location of the maximum correlation. Default is False.

    Returns:
    - corr (numpy.ndarray): The cross-correlation result.
    - x (int): The x-coordinate of the maximum correlation (if return_max_loc=True).
    - y (int): The y-coordinate of the maximum correlation (if return_max_loc=True).
    """
    corr = signal.correlate2d(filter, background_image, boundary='symm', mode='same')
    if return_max_loc:
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        return corr, x, y
    return corr

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
    padded_image = np.zeros(target_shape)
    
    offset_x = (target_shape[1] - current_shape[1]) // 2
    offset_y = (target_shape[0] - current_shape[0]) // 2
    
    padded_image[offset_y:offset_y+current_shape[0], offset_x:offset_x+current_shape[1]] = image
    return padded_image

def rotational_cc(background_image, filter, angle=10, return_max_loc=False, return_average=False):
    """
    Perform rotational cross-correlation between a background image and a filter image.

    Parameters:
    - background_image (numpy.ndarray): The background image.
    - filter (numpy.ndarray): The filter image.
    - angle (int, optional): The angle by which to rotate the filter image for each iteration. Default is 10.
    - return_max_loc (bool, optional): Whether to return the location of the maximum correlation. Default is False.
    - return_average (bool, optional): Whether to return the average correlation result. Default is False.

    Returns:
    - correlation_matrices (list): List of correlation matrices (if return_average=False).
    - max_locs (list): List of maximum correlation locations [(x1, y1), (x2, y2), ...] (if return_max_loc=True).
    - average_corr (numpy.ndarray): The average cross-correlation result (if return_average=True).
    - max_corr (float): The maximum correlation value (if return_max_loc=True and return_average=False).
    - best_rotation (int): The rotation angle with the highest correlation value (if return_max_loc=True and return_average=False).
    """
    assert angle < 360
    filter_angles = np.arange(0, 360, angle)
    
    max_corr = None
    best_rotation = None
    best_loc = None
    all_correlations = []
    
    target_shape = filter.shape
    
    for filter_angle in filter_angles:
        rotated_filter = rotate(filter, filter_angle, reshape=False)
        rotated_filter = pad_or_crop_image(rotated_filter, target_shape)
        
        corr, x, y = cross_correlation(background_image, rotated_filter, return_max_loc=True)
        
        if return_average:
            all_correlations.append(corr)
        
        if max_corr is None or np.max(corr) > max_corr:
            max_corr = np.max(corr)
            best_rotation = filter_angle
            best_loc = (x, y)
    
    if return_average:
        average_corr = np.mean(all_correlations, axis=0)
        if return_max_loc:
            return average_corr, max_corr, best_rotation, best_loc
        else:
            return average_corr
    
    if return_max_loc:
        return max_corr, best_rotation, best_loc
    else:
        return max_corr


def plot_3d_topography(data, elev=30, azim=45, save_path=None):
    """
    Plot a 3D topographical representation of data.

    Parameters:
    - data (numpy.ndarray): The data to be plotted.
    - elev (int, optional): The elevation angle in degrees. Default is 30.
    - azim (int, optional): The azimuth angle in degrees. Default is 45.
    - save_path (str, optional): The file path to save the plot as HTML. Default is None.

    Returns:
    None
    """
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    z = data

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(scene=dict(xaxis_title='X',
                                 yaxis_title='Y',
                                 zaxis_title='Correlation Value',
                                 aspectmode='manual',
                                 aspectratio=dict(x=1, y=1, z=0.5)),
                      scene_camera=dict(eye=dict(x=0, y=0, z=1.5),
                                        up=dict(x=0, y=0, z=1)),
                      )

    if save_path:
        pio.write_html(fig, save_path)
        print(f"Interactive plot saved to: {save_path}")
    else:
        fig.show()
    x = np.arange(data.shape[1])
    y = np.arange(data.shape[0])
    x, y = np.meshgrid(x, y)
    z = data

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(scene=dict(xaxis_title='X',
                                 yaxis_title='Y',
                                 zaxis_title='Correlation Value',
                                 aspectmode='manual',
                                 aspectratio=dict(x=1, y=1, z=0.5)),
                      scene_camera=dict(eye=dict(x=0, y=0, z=1.5),
                                        up=dict(x=0, y=0, z=1)))

    if save_path:
        pio.write_html(fig, save_path)
        print(f"Interactive plot saved to: {save_path}")
    else:
        fig.show()


if __name__ == "__main__":
    # Example usage
    background_image = np.random.rand(100, 100)
    filter = np.random.rand(20, 20)
    # background_image, filter = scipy_example()

    average_corr = rotational_cc(background_image, filter, angle=10, return_average=True)

    # Save interactive plot to a file
    plot_3d_topography(average_corr, save_path='3d_topography_plot.html')
