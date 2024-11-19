# main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, util
from skimage.util import img_as_float
import os

# Import helper functions
from helpers.filters import gaussian_kernel, laplacian_of_gaussian
from helpers.normalization import normalize_LOG_display
from helpers.edge_detection import detect_zero_crossings, non_maximum_suppression, double_threshold
from helpers.visualization import display_images


def main():
    # Clear all plots
    plt.close('all')

    # ==============================================
    #       Load and Preprocess Image
    # ==============================================

    # Ensure the image exists
    image_path = 'graylevel.jpg'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file '{image_path}' does not exist in the current directory.")

    # Read the image
    I = io.imread(image_path)

    # Convert to grayscale if it's a color image
    if I.ndim == 3:
        I = color.rgb2gray(I)

    # Convert to float (0.0 to 255.0)
    I = img_as_float(I) * 255.0

    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(I, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    plt.show()

    # ==============================================
    # Part I: Edge Detection using Multiscale LOG Detectors
    # ==============================================

    print('--- Part I: Multiscale LOG Edge Detection ---')

    # Define scales for multiscale approach
    sigma_values = [2, 8, 16]

    # Task 1: Blur the image with Gaussian filters at different scales
    print('Task 1: Blurring the image with Gaussian filters at different scales')

    blurred_images = []
    blurred_titles = []
    for sigma in sigma_values:
        G = gaussian_kernel(sigma)
        I_blur = ndimage.convolve(I, G, mode='reflect')
        blurred_images.append(I_blur.astype(np.uint8))
        blurred_titles.append(f'Gaussian Blur σ={sigma}')

    display_images(blurred_images, blurred_titles, figsize=(15, 5), suptitle='Part I - Task 1: Gaussian Blurs')

    # Task 2: Apply LOG filters at the chosen scales to detect edges
    print('Task 2: Applying LOG filters to detect edges')

    for sigma in sigma_values:
        LOG = laplacian_of_gaussian(sigma)
        I_LOG = ndimage.convolve(I, LOG, mode='reflect')

        # a) Positive and negative values with zero as gray (128)
        I_a_display = normalize_LOG_display(I_LOG, 'gray')

        # b) Positive in white (255) and negative in black (0)
        I_b_display = normalize_LOG_display(I_LOG, 'binary')

        # c) Zero-crossings indicating edges
        zero_cross = detect_zero_crossings(I_LOG)
        I_c_display = (zero_cross * 255).astype(np.uint8)

        # Display the results
        images = [I_a_display, I_b_display, I_c_display]
        titles = ['Positive & Negative (Gray=128)', 'Positive=White, Negative=Black', 'Zero-Crossings (Edges)']
        suptitle = f'LOG Filter Results σ={sigma}'
        display_images(images, titles, figsize=(18, 6), suptitle=suptitle)

    # Task 3: Add Gaussian noise and evaluate edge detection performance
    print('Task 3: Adding Gaussian noise and evaluating LOG edge detection')

    # Define noise levels and corresponding standard deviations
    noise_levels = ['Light', 'Moderate', 'Heavy']
    sigma_noise = [5, 15, 30]

    for level, sigma_n in zip(noise_levels, sigma_noise):
        # Add Gaussian noise
        I_noisy = I + sigma_n * np.random.randn(*I.shape)
        I_noisy = np.clip(I_noisy, 0, 255)  # Clip to [0, 255]

        plt.figure(figsize=(15, 5))
        for idx, sigma in enumerate(sigma_values):
            LOG = laplacian_of_gaussian(sigma)
            I_LOG_noisy = ndimage.convolve(I_noisy, LOG, mode='reflect')

            # Zero-crossings
            zero_cross_noisy = detect_zero_crossings(I_LOG_noisy)
            I_zero_cross_noisy = (zero_cross_noisy * 255).astype(np.uint8)

            # Display detected edges
            plt.subplot(len(sigma_values), 1, idx + 1)
            plt.imshow(I_zero_cross_noisy, cmap='gray')
            plt.title(f'Detected Edges σ={sigma}')
            plt.axis('off')

        plt.suptitle(f'LOG Edge Detection with {level} Gaussian Noise (σ_noise={sigma_n})')
        plt.tight_layout()
        plt.show()

    # ==============================================
    # Part II: Edge Detection using Canny Edge Detectors
    # ==============================================

    print('--- Part II: Canny Edge Detection ---')

    # Task 1: Blur the image with Gaussian filters at different scales
    print('Task 1: Blurring the image with Gaussian filters at different scales')

    blurred_images = []
    blurred_titles = []
    for sigma in sigma_values:
        G = gaussian_kernel(sigma)
        I_blur = ndimage.convolve(I, G, mode='reflect')
        blurred_images.append(I_blur.astype(np.uint8))
        blurred_titles.append(f'Gaussian Blur σ={sigma}')

    display_images(blurred_images, blurred_titles, figsize=(15, 5), suptitle='Part II - Task 1: Gaussian Blurs')

    # Task 2: Apply Canny edge detectors at the chosen scales to detect edges
    print('Task 2: Applying Canny edge detectors to detect edges')

    # Define Sobel filters
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = Gx.T

    for sigma in sigma_values:
        G = gaussian_kernel(sigma)
        I_blur = ndimage.convolve(I, G, mode='reflect')

        # a) Gradient magnitude and angle
        Ix = ndimage.convolve(I_blur, Gx, mode='reflect')
        Iy = ndimage.convolve(I_blur, Gy, mode='reflect')
        Gmag = np.hypot(Ix, Iy)
        Gmag = (Gmag / Gmag.max()) * 255  # Normalize to [0,255]
        Gdir = np.degrees(np.arctan2(Iy, Ix))

        # b) Non-Maximum Suppression
        nms = non_maximum_suppression(Gmag, Gdir)

        # c) Double Thresholding and Edge Linking
        high_thresh = Gmag.max() * 0.3
        low_thresh = high_thresh * 0.5
        edges = double_threshold(nms, low_thresh, high_thresh)

        # Display the results
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(I_blur.astype(np.uint8), cmap='gray')
        plt.title(f'Blurred Image σ={sigma}')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(Gmag, cmap='gray')
        plt.title('Gradient Magnitude')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(nms, cmap='gray')
        plt.title('After Non-Maximum Suppression')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(edges, cmap='gray')
        plt.title('Final Edge Map')
        plt.axis('off')

        plt.suptitle(f'Canny Edge Detection Results σ={sigma}')
        plt.show()

    # Task 3: Add Gaussian noise and evaluate edge detection performance
    print('Task 3: Adding Gaussian noise and evaluating Canny edge detection')

    for level, sigma_n in zip(noise_levels, sigma_noise):
        # Add Gaussian noise
        I_noisy = I + sigma_n * np.random.randn(*I.shape)
        I_noisy = np.clip(I_noisy, 0, 255)  # Clip to [0, 255]

        plt.figure(figsize=(15, 5))
        for idx, sigma in enumerate(sigma_values):
            G = gaussian_kernel(sigma)
            I_blur = ndimage.convolve(I_noisy, G, mode='reflect')

            # Gradient calculation
            Ix = ndimage.convolve(I_blur, Gx, mode='reflect')
            Iy = ndimage.convolve(I_blur, Gy, mode='reflect')
            Gmag = np.hypot(Ix, Iy)
            Gmag = (Gmag / Gmag.max()) * 255  # Normalize to [0,255]
            Gdir = np.degrees(np.arctan2(Iy, Ix))

            # Non-Maximum Suppression
            nms = non_maximum_suppression(Gmag, Gdir)

            # Double Thresholding and Edge Linking
            high_thresh = Gmag.max() * 0.3
            low_thresh = high_thresh * 0.5
            edges = double_threshold(nms, low_thresh, high_thresh)

            # Display detected edges
            plt.subplot(len(sigma_values), 1, idx + 1)
            plt.imshow(edges, cmap='gray')
            plt.title(f'Detected Edges σ={sigma}')
            plt.axis('off')

        plt.suptitle(f'Canny Edge Detection with {level} Gaussian Noise (σ_noise={sigma_n})')
        plt.tight_layout()
        plt.show()

    # ==============================================
    #             End of Program
    # ==============================================
    print('Edge detection tasks completed successfully.')


if __name__ == "__main__":
    main()
