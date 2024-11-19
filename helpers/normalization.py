# helpers/normalization.py

import numpy as np

def normalize_LOG_display(I_LOG, mode):
    """
    Normalize LOG-filtered image for display purposes

    Parameters:
        I_LOG: The LOG-filtered image
        mode: 'gray' or 'binary'
        'gray': Positive and negative values with zero at 128
        'binary': Positive values as white (255), negative as black (0)

    Returns:
        I_display: The normalized image ready for display
    """
    if mode == 'gray':
        min_val = np.min(I_LOG)
        max_val = np.max(I_LOG)
        I_a_normalized = (I_LOG - min_val) / (max_val - min_val) * 255
        I_display = I_a_normalized.astype(np.uint8)
        # Shift to have zero at 128
        I_display = np.clip(I_display - 127.5 + 128, 0, 255).astype(np.uint8)
    elif mode == 'binary':
        I_display = np.where(I_LOG > 0, 255, 0).astype(np.uint8)
    else:
        raise ValueError('Unknown mode for normalize_LOG_display.')
    return I_display
