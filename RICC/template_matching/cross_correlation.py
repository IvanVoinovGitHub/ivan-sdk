import numpy as np
from scipy import signal
from scipy import datasets
from scipy.ndimage import rotate
from visualization import plot_max_cross_correlation, plot_3d_topography
from image_utils import normalize_image, noise
from joblib import Parallel, delayed


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
        # Add Gaussian noise to the background image
        background_image = noise(background_image)

    return background_image, filter


def cross_correlation(background_image, filter, return_max_loc=False, normalize=False):
    """
    Compute the cross-correlation between a background image and a filter image.

    Parameters:
    - background_image (numpy.ndarray): The background image.
    - filter (numpy.ndarray): The filter image.
    - return_max_loc (bool, optional): Whether to return the location of the maximum correlation. Default is False.
    - normalize (bool, optional): Whether to normalize the background_image and filter. Default is False.
    Returns:
    - corr (numpy.ndarray): The cross-correlation result.
    - x (int): The x-coordinate of the maximum correlation (if return_max_loc=True).
    - y (int): The y-coordinate of the maximum correlation (if return_max_loc=True).
    """
    if normalize:
        background_image, filter = normalize_image(background_image), normalize_image(filter)

    corr = signal.correlate2d(background_image, filter, boundary='symm', mode='same')
    if return_max_loc:
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        return corr, (x, y)
    return corr, None


def rotational_cc(background_image, filter, angle=10, use_normalized=True, return_average=False, return_max=False, n_jobs=-1):
    """
    Compute the cross-correlation of a filter rotated by a certain angle over a background image.
    
    Args:
    background_image (numpy.ndarray): The larger image.
    filter (numpy.ndarray): The smaller image (filter).
    angle (int): The angle increment for rotating the filter.
    use_normalized (bool): Whether to use normalized cross-correlation.
    return_average (bool): Whether to return the average of the cross-correlation matrices.
    return_max (bool): Whether to return the maximum of the cross-correlation matrices.
    n_jobs (int): The number of jobs to run in parallel. -1 means using all processors.

    Returns:
    numpy.ndarray: The resulting cross-correlation matrices or the average/max of them.
    list: The (x, y) locations of the maximum values for each rotation if return_max_loc is True.
    """
    assert angle < 360
    angles = np.arange(0, 360, angle)

    if use_normalized:
        background_image, filter = normalize_image(background_image), normalize_image(filter)

    def process_angle(filter_angle):
        rotated_filter = rotate(filter, filter_angle, reshape=True)
        return cross_correlation(background_image, rotated_filter, return_max_loc=True)


    results = Parallel(n_jobs=n_jobs)(delayed(process_angle)(filter_angle) for filter_angle in angles)

    all_correlations, max_locs = zip(*results)
    all_correlations = np.array(all_correlations)

    if return_average and return_max:
        return np.mean(all_correlations, axis=0), np.max(all_correlations, axis=0)
    elif return_average:
        return np.mean(all_correlations, axis=0)
    elif return_max:
        return np.max(all_correlations, axis=0)
    
    return all_correlations, max_locs


def threshold_correlation(correlation, threshold=0.8):
    """
    Apply a threshold to a correlation matrix, setting values below the threshold to zero.

    Parameters:
    - correlation (numpy.ndarray): The input correlation matrix.
    - threshold (float, optional): The threshold value. Default is 0.8.

    Returns:
    - correlation (numpy.ndarray): The thresholded correlation matrix.
    """
    correlation[correlation < threshold] = 0
    return correlation


if __name__ == "__main__":
    # Example usage
    background_image = np.random.rand(100, 100)
    filter = np.random.rand(20, 20)

    corr, max_loc = cross_correlation(background_image, filter, return_max_loc=True)
    plot_max_cross_correlation(background_image, filter, corr, max_loc, save_path='2d_images.png')

    average_corr, max_corr = rotational_cc(background_image, filter, angle=10, return_average=True, return_max=True)

    # Save interactive plot to a file
    plot_3d_topography(average_corr, save_path='3d_avg_topography_plot.html')
    plot_3d_topography(max_corr, save_path='3d_max_topography_plot.html')
