# helpers/edge_detection.py

import numpy as np


def detect_zero_crossings(I_LOG, threshold=10):
    """
    Detect zero-crossings in LOG-filtered image

    Parameters:
        I_LOG: The LOG-filtered image
        threshold: Minimum absolute value to consider a zero-crossing

    Returns:
        zero_cross: Boolean array indicating zero-crossings
    """
    zero_cross = np.zeros(I_LOG.shape, dtype=bool)
    # Define shifts: right, left, up, down, upper-right, upper-left, lower-right, lower-left
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, 1), (-1, -1), (1, 1), (1, -1)]

    for dx, dy in shifts:
        shifted = np.roll(np.roll(I_LOG, dx, axis=0), dy, axis=1)
        # To avoid circular shift artifacts, set border regions to False
        if dx > 0:
            shifted[:dx, :] = 0
        elif dx < 0:
            shifted[dx:, :] = 0
        if dy > 0:
            shifted[:, :dy] = 0
        elif dy < 0:
            shifted[:, dy:] = 0
        zero_cross |= (I_LOG * shifted) < 0

    # Apply threshold
    zero_cross &= (np.abs(I_LOG) > threshold)
    return zero_cross


def non_maximum_suppression(Gmag, Gdir):
    """
    Perform non-maximum suppression on gradient magnitude and direction

    Parameters:
        Gmag: Gradient magnitude
        Gdir: Gradient direction in degrees

    Returns:
        nms: Non-maximum suppressed image
    """
    rows, cols = Gmag.shape
    nms = np.zeros((rows, cols), dtype=np.float32)
    angle = Gdir.copy()
    angle[angle < 0] += 180  # Map angles to [0,180]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = Gmag[i, j + 1]
                    r = Gmag[i, j - 1]
                # Angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = Gmag[i + 1, j - 1]
                    r = Gmag[i - 1, j + 1]
                # Angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = Gmag[i + 1, j]
                    r = Gmag[i - 1, j]
                # Angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = Gmag[i - 1, j - 1]
                    r = Gmag[i + 1, j + 1]

                if (Gmag[i, j] >= q) and (Gmag[i, j] >= r):
                    nms[i, j] = Gmag[i, j]
                else:
                    nms[i, j] = 0
            except IndexError:
                pass

    return nms


def double_threshold(nms, low_thresh, high_thresh):
    """
    Perform double thresholding and edge linking (hysteresis)

    Parameters:
        nms: Non-maximum suppressed image
        low_thresh: Low threshold value
        high_thresh: High threshold value

    Returns:
        edges: Binary edge map
    """
    strong = nms > high_thresh
    weak = (nms >= low_thresh) & (nms <= high_thresh)
    edges = np.zeros(nms.shape, dtype=np.uint8)
    edges[strong] = 1

    # Edge tracking by hysteresis
    rows, cols = edges.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak[i, j]:
                if np.any(edges[i - 1:i + 2, j - 1:j + 2]):
                    edges[i, j] = 1

    return edges
