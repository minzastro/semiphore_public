import numpy as np
from astropy.stats import sigma_clip


def filter_data(zdata, sigma=4):
    """Create sigma-clipping and no NaNs mask from the data.

    Args:
        zdata: input data
        column_count: number of columns to work on

    Returns:
        [bool]: mask to apply
    """
    column_count = zdata.shape[1]
    mask = np.ones(len(zdata), dtype=bool)
    for c in range(column_count - 1):
        mask1 = sigma_clip(zdata[:, c] - zdata[:, c+1],
                           sigma=sigma).mask
        mask2 = np.isnan(np.array(zdata[:, c] - zdata[:, c + 1]))
        mask *= ~mask1 + mask2
    return mask
