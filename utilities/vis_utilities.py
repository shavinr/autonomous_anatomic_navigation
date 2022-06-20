import numpy as np


def transparent_cmap(color_map):
    """Copy colormap and set alpha values

    Parameters
    ----------
    color_map: Matplotlib colormap object

    Returns
    -------
    Colormap modified with alpha axis with transparency.
    """
    color_map._init()
    color_map._lut[:, -1] = np.linspace(0, 0.8, len(color_map._lut[:, -1]))
    return color_map