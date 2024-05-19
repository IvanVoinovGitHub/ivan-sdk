import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import datasets
from scipy.ndimage import rotate
import plotly.graph_objs as go
import plotly.io as pio


def scipy_example(normalize=True, add_noise=True):
    """
    Generates an example background image and filter from the scipy built-in dataset.

    Parameters:
    normalize (bool): If True, normalizes the images by subtracting the mean intensity.
    add_noise (bool): If True, adds Gaussian noise to the background image.

    Returns:
    tuple: A tuple containing:
        - background_image (numpy.ndarray): The generated background image.
        - filter (numpy.ndarray): The filter (a sub-region of the background image).
    """
    if normalize:
        # Load the grayscale face image and normalize it by subtracting the mean
        background_image = datasets.face(gray=True) - datasets.face(gray=True).mean()
        # Extract a sub-region of the face image (right eye) to use as the filter and normalize it by subtracting the mean
        filter = np.copy(background_image[300:365, 670:750])
        filter -= filter.mean()
    else:
        # Load the grayscale face image without normalization
        background_image = datasets.face(gray=True)
        # Extract a sub-region of the face image (right eye) to use as the filter
        filter = np.copy(background_image[300:365, 670:750])
    
    if add_noise:
        # Initialize random number generator
        rng = np.random.default_rng()
        # Add Gaussian noise to the background image
        background_image = background_image + rng.standard_normal(background_image.shape) * 50

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
    corr = signal.correlate2d(background_image, filter, boundary='symm', mode='same')
    if return_max_loc:
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        return corr, (x, y)
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


def rotational_cc(background_image, filter, angle=10, return_average=False, return_max=False):
    """
    Compute the cross-correlation of a filter rotated by a certain angle over a background image.
    
    Args:
    background_image (numpy.ndarray): The larger image.
    filter (numpy.ndarray): The smaller image (filter).
    angle (int): The angle increment for rotating the filter.
    return_average (bool): Whether to return the average of the cross-correlation matrices.
    return_max (bool): Whether to return the maximum of the cross-correlation matrices.

    Returns:
    numpy.ndarray: The resulting cross-correlation matrices.
    list: The (x, y) locations of the maximum values for each rotation.
    """
    assert angle < 360
    filter_angle = 0
    all_correlations = []
    max_locs = []

    while filter_angle < 360:
        rotated_filter = rotate(filter, filter_angle, reshape=True)
        corr, max_loc = cross_correlation(background_image, rotated_filter, return_max_loc=True)
        all_correlations.append(corr)
        max_locs.append(max_loc)
        filter_angle += angle

    if return_average:
        return np.mean(all_correlations, axis=0)
    if return_max:
        return np.max(all_correlations, axis=0)
    
    return all_correlations, max_locs


def plot_max_cross_correlation(background_image, filter, corr, max_loc, save_path=None):
    """
    Display and optionally save the original image, filter image, and their cross-correlation result.

    Parameters:
    - background_image (numpy.ndarray): The background image.
    - filter (numpy.ndarray): The filter image.
    - corr (numpy.ndarray): The cross-correlation result.
    - max_loc (tuple): The (x, y) coordinates of the maximum correlation.
    - save_path (str, optional): The file path to save the figure. If None, the figure is not saved. Default is None.

    Returns:
    None
    """
    fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(6, 15))
    
    # Original image
    ax_orig.imshow(background_image, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    
    # Filter image
    ax_template.imshow(filter, cmap='gray')
    ax_template.set_title('Filter')
    ax_template.set_axis_off()
    
    # Cross-correlation result
    ax_corr.imshow(corr, cmap='gray')
    ax_corr.set_title('Cross-correlation')
    ax_corr.set_axis_off()
    
    # Highlight the maximum correlation location on the original image
    ax_orig.plot(max_loc[0], max_loc[1], 'ro') 
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to: {save_path}")
    else:
        # Show the figure
        plt.show()


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

    corr, max_loc = cross_correlation(background_image, filter, return_max_loc=True)
    plot_max_cross_correlation(background_image, filter, corr, max_loc, save_path='2d_images.png')

    average_corr = rotational_cc(background_image, filter, angle=10, return_average=True)
    max_corr = rotational_cc(background_image, filter, angle=10, return_max=True)

    # Save interactive plot to a file
    plot_3d_topography(average_corr, save_path='3d_avg_topography_plot.html')
    plot_3d_topography(max_corr, save_path='3d_max_topography_plot.html')
