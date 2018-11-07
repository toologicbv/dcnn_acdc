import numpy as np


def transparent_cmap(cmap, N=255, alpha=0.8):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, alpha, N + 4)
    return mycmap
