****Edge Detection Project****
This repository provides a comprehensive implementation of edge detection algorithms using Python, focusing on both Multiscale Laplacian of Gaussian (LOG) and Canny Edge Detectors. The project is organized into modular components to enhance readability, maintainability, and scalability.

**Introduction**
Edge detection is a fundamental tool in image processing, computer vision, and machine learning. It involves identifying points in a digital image where the image brightness changes sharply, indicating object boundaries and features. This project explores two prominent edge detection techniques:

- Multiscale Laplacian of Gaussian (LOG) Edge Detection
- Canny Edge Detection
  
Additionally, the project evaluates the robustness of these methods by introducing Gaussian noise at varying levels.

**Features**
Modular Design: Organized into separate modules for filters, normalization, edge detection, and visualization.
Multiscale LOG Edge Detection: Applies LOG filters at multiple scales to detect edges.
Canny Edge Detection: Implements the Canny algorithm step-by-step, including gradient calculation, non-maximum suppression, and double thresholding.
Noise Evaluation: Adds Gaussian noise to images and assesses the performance of edge detectors under noisy conditions.
Visualization: Comprehensive visualization of intermediate and final results for better understanding and analysis.
