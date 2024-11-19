# helpers/visualization.py

import matplotlib.pyplot as plt

def display_images(images, titles, figsize=(15, 5), suptitle=None):
    """
    Display multiple images in a single row

    Parameters:
        images: List of images to display
        titles: List of titles for each subplot
        figsize: Figure size
        suptitle: Overall title for the figure
    """
    num_images = len(images)
    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis('off')
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()

def display_edge_map(edge_map, title, figsize=(6,6)):
    """
    Display a single edge map

    Parameters:
        edge_map: Binary edge map to display
        title: Title for the subplot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(edge_map, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
