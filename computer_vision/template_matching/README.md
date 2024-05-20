# 2D Cross-Correlation Visualization

[Link to 3d Plot](https://ivanvoinovgithub.github.io/ivan-sdk/)

This project demonstrates the process of 2D cross-correlation using images. The repository includes code to compute the cross-correlation of a template image with a background image, as well as to visualize this process step-by-step in a GIF. Furthermore, this repository includes an implementation of rotational cross-correlation where the cross correlation is computed iteratively over rotated variants of the template images. Upon doing so, the stack of cross correlation outputs has an aggregate operation of mean and/or max applied. To visualize the results, an interative 3d Topographical representation may be created.

## What is 2D Cross-Correlation?

2D cross-correlation is a mathematical operation that measures the similarity between a template and regions of a larger image. It involves sliding the template over the background image and computing a correlation value at each position. The output is a 2D array where each value represents the similarity between the template and the corresponding region of the background image.

### Mathematical Formulation

Given a background image \(I\) of size \(M \times N\) and a template \(T\) of size \(m \times n\), the 2D cross-correlation \(C\) is defined as:

\[ C(i, j) = \sum_{u=0}^{m-1} \sum_{v=0}^{n-1} I(i+u, j+v) \cdot T(u, v) \]

where \( (i, j) \) are the coordinates of the top-left corner of the region in the background image being compared with the template.

## Example

Here, we use a simple example to illustrate 2D cross-correlation.

### Cross-Correlation

![Cross-Correlation](figures/scipy/2d_downsampled_images.png)

### Cross-Correlation Process GIF

The following GIF shows the template sliding over the background image, with the cross-correlation values being updated.

![Cross-Correlation Process](figures/scipy/correlation_process.gif)

