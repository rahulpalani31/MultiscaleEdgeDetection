# helpers/filters.py

import numpy as np

def gaussian_kernel(sigma):
    """
    Create a Gaussian kernel given sigma
    """
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Ensures coverage up to ±3σ
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def laplacian_of_gaussian(sigma):
    """
    Create a Laplacian of Gaussian (LOG) kernel given sigma
    """
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    normalization = (xx**2 + yy**2 - 2 * sigma**2) / (sigma**4)
    G = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    LOG = normalization * G
    LOG -= np.mean(LOG)  # Ensure zero mean
    return LOG
