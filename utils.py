import logging
import io

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image


class DotDict(dict):
    """A dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(model, path, loss, cost):
    name = type(model).__name__
    filename = f"{name}-loss-{loss:.2f}-cost-{cost:.2f}.pth.tar"
    torch.save(model.state_dict(), path/filename)


def input_output_img(target, output):
    """Plot target and output sequences of bit vectors.
    One under the other. Matplotlib figure size is tweaked
    best for sequence legth in [20, 40, 100].

    Input:
      target (np.array)
      output (np.array)  (same shape as target)
    Returns:
      matplotlib.figure.Figure
    """

    fig_shape = int(0.15 * target.shape[1])
    fig = plt.figure(figsize=(fig_shape, fig_shape))
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(2, 1),
        axes_pad=0.3,
        add_all=True,
        aspect=True,
        label_mode="L",
        cbar_mode="single",
        cbar_size=0.15,
        cbar_pad=0.1
    )
    plt.axis('off')

    grid[0].imshow(target, cmap='jet')
    grid[0].axis('off')
    grid[0].set_title('Target:', loc='left')

    im = grid[1].imshow(output, cmap='jet')
    grid[1].axis('off')
    grid[1].set_title('Output:', loc='left')
    grid.cbar_axes[0].colorbar(im)

    buff = io.BytesIO()
    fig.savefig(buff, bbox_inches='tight', pad_inches=0.1, dpi=90)
    buff.seek(0)
    img = np.rollaxis(np.asarray(Image.open(buff)), -1, 0)
    plt.close(fig)
    return img
