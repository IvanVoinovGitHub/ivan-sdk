import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy import signal
from image_utils import pad_or_crop_image


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


def normalize_array(arr, new_min=0, new_max=255):
    """Normalize a NumPy array to a new range [new_min, new_max]."""
    arr_min, arr_max = arr.min(), arr.max()
    norm_arr = (arr - arr_min) * (new_max - new_min) / (arr_max - arr_min) + new_min
    return norm_arr


def create_gif_of_correlation(background_image, filter, save_path='correlation_process.gif', step=1, pause_duration=500):
    """
    Creates a GIF showing the process of 2D cross-correlation.

    Parameters:
    background_image (numpy.ndarray): The background image.
    filter (numpy.ndarray): The template to slide across the background image.
    save_path (str): Path to save the generated GIF.
    step (int): The step size for sliding the template. Larger steps reduce the number of frames.
    pause_duration (int): Duration in milliseconds to pause at the end of the GIF.

    Returns:
    None
    """
    # Get the dimensions of the background image and template
    bg_height, bg_width = background_image.shape
    temp_height, temp_width = filter.shape
    bg_height, bg_width = bg_height + 2 * (temp_height - 1), bg_width + 2 * (temp_width - 1)
    new_shape = (bg_height, bg_width)

    # Compute the cross-correlation
    correlation = signal.correlate2d(background_image, filter, boundary='symm', mode='same')

    # Normalize correlation to the same range as the background image
    correlation = normalize_array(correlation, new_min=background_image.min(), new_max=background_image.max())

    # Pad Background Image
    background_image = pad_or_crop_image(background_image, new_shape)
    
    frames = []
    
    # Create each frame of the GIF
    for y in range(0, bg_height - temp_height, step):
        for x in range(0, bg_width - temp_width, step):
            fig, ax = plt.subplots()

            # Update the relevant portion of the background image with cross-correlation values
            if x < bg_width - 2 * temp_width and y < bg_height - 2 * temp_height:
                sub_corr = correlation[0:y, 0:x]
                background_image[temp_height:y+temp_height, temp_width:x+temp_width] = sub_corr
            updated_image = np.copy(background_image)
            updated_image[y:y+temp_height, x:x+temp_width] = filter
            
            ax.imshow(updated_image, cmap='gray')

            # Draw a red rectangle to show the template position
            rect = patches.Rectangle((x-1, y-1), temp_width, temp_height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            ax.set_axis_off()

            # Save the frame to a PIL Image
            fig.canvas.draw()
            buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buffer = buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame = Image.fromarray(buffer)
            frames.append(frame)
            plt.close(fig)
    
    # Duplicate the last frame to create a pause at the end
    for _ in range(pause_duration // 100):  # Adjust number of duplicates based on pause duration
        frames.append(frames[-1])
                      
    # Save the frames as a GIF
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=10, loop=0)

