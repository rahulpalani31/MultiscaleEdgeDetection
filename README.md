# Edge Detection Project

This repository provides a comprehensive implementation of edge detection algorithms using Python, focusing on both Multiscale Laplacian of Gaussian (LOG) and Canny Edge Detectors. The project is organized into modular components to enhance readability, maintainability, and scalability.

# Features

**Modular Design:** Organized into separate modules for filters, normalization, edge detection, and visualization.

**Multiscale LOG Edge Detection:** Applies LOG filters at multiple scales to detect edges.

**Canny Edge Detection:** Implements the Canny algorithm step-by-step, including gradient calculation, non-maximum suppression, and double thresholding.

**Noise Evaluation:** Adds Gaussian noise to images and assesses the performance of edge detectors under noisy conditions.

**Visualization:** Comprehensive visualization of intermediate and final results for better understanding and analysis.


# Project Overview

The script will sequentially perform the following:

Load and Preprocess the Image: Reads graylevel.jpg, converts it to grayscale (if necessary), and displays it.

**Part I:** Multiscale LOG Edge Detection:

- Task 1: Applies Gaussian blurs at multiple scales and displays the blurred images.
- Task 2: Applies LOG filters at chosen scales, visualizes different representations (positive & negative values, binary representation, zero-crossings), and displays the results.
- Task 3: Adds Gaussian noise at varying levels, applies LOG edge detection, and visualizes the detected edges under noisy conditions.

**Part II:** Canny Edge Detection:
  
- Task 1: Applies Gaussian blurs at multiple scales and displays the blurred images.
- Task 2: Implements the Canny edge detection algorithm step-by-step (gradient calculation, non-maximum suppression, double thresholding) and displays intermediate and final results.
- Task 3: Adds Gaussian noise at varying levels, applies Canny edge detection, and visualizes the detected edges under noisy conditions.

Completion Message: Prints a confirmation message upon successful completion of all tasks.

# Understanding the Workflow

**Gaussian Blurring:** Smooths the image to reduce noise and detail, controlled by the sigma value. Higher sigma values result in more blurring.

**LOG Filtering:** Enhances edges by applying the Laplacian of Gaussian filter, which highlights regions of rapid intensity change.

**Zero-Crossings Detection:** Identifies edges by finding points where the LOG-filtered image changes sign.

**Canny Edge Detection:**

- **Gradient Calculation:** Computes the intensity gradients using the Sobel operator.
- **Non-Maximum Suppression:** Thins edges to retain only the local maxima in the gradient direction.
- **Double Thresholding:** Differentiates between strong and weak edges.
- **Edge Linking (Hysteresis):** Connects weak edges to strong edges if they are adjacent, forming continuous edges.

**Noise Evaluation:** Simulates real-world scenarios by introducing Gaussian noise at varying levels and assesses the robustness of edge detection methods.
