import logging
import argparse
import pathlib
import shutil
import io

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def read_config():
    parser = argparse.ArgumentParser(
        prog='Train/Eval script',
        description=('Script for training and evaluating memory models on various bitmap tasks. '
                     'All parameters should be given throug the config file.'),
    )
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        required=True,
        help='Name of the current experiment. Can also provide name/with/path for grouping'
    )
    parser.add_argument(
        '-k', '--keep',
        action='store_true',
        help='Keep logs from previous run.'
    )
    parser.add_argument(
        '-l', '--load',
        help='Path to model checkpoint file',
        default=None,
    )

    args = parser.parse_args()
    path = pathlib.Path('experiments')/args.name

    assert (path/'config.yaml').exists(), 'No configuration file found.'

    with open(path/'config.yaml') as f:
        config = DotDict(yaml.safe_load(f))

    # clear previous run logs
    if not args.keep:
        (path/'tensorboard').exists() and shutil.rmtree(path/'tensorboard')
        if args.load is None:
            (path/'checkpoints').exists() and shutil.rmtree(path/'checkpoints')
        open(path/'train.log', 'w').close()

    config.path = path
    config.load = args.load
    return config


def save_checkpoint(
        model, optimizer, step,
        train_data,
        path,
):
    name = type(model).__name__
    filename = f"{name}-{step}.pth"
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'task_random_state': train_data.rand.get_state(),
            'step': step,
        },
        path/filename
    )


def load_checkpoint(
        model, optimizer,
        train_data,
        path,
):
    try:
        checkpoint = torch.load(path)
    except RuntimeError:
        checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_data.rand.set_state(checkpoint['task_random_state'])
    step = checkpoint['step']

    return model, optimizer, train_data, step


def fig2img(fig):
    buff = io.BytesIO()
    fig.savefig(buff, bbox_inches='tight', pad_inches=0.1, dpi=90)
    buff.seek(0)
    img = np.rollaxis(np.asarray(Image.open(buff)), -1, 0)
    plt.close(fig)

    return img


def input_output_img(
        target,
        output,
        text_top='Target',
        text_bottom='Output',
):
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

    return fig2img(fig)


def dnc_img(info):
    def add_subplot(subplot, image, title, ticks):
        plt.subplot(subplot)
        im = plt.imshow(image, aspect='auto')
        plt.title(title, loc='left')
        plt.axis('on')
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        return im

    # Write info
    write_weights = np.array(info['write_head']['write_weights']).T, 'Write weights'
    alloc_gates = np.array(info['write_head']['alloc_gate']).reshape(1, -1), 'Allocation gates'
    write_gates = np.array(info['write_head']['write_gate']).reshape(1, -1), 'Write gates'

    # Read info
    read_modes = np.array(info['read_head']['read_modes'])
    read_weights = np.array(info['read_head']['read_weights']).T, 'Read weights'
    forward_modes = read_modes[:, 0:1].T, 'Forward mode'
    backward_modes = read_modes[:, 1:2].T, 'Backward mode'
    content_modes = read_modes[:, 2:].T, 'Content mode'

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    plt.gray()
    grid = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)
    write = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=grid[0],
        height_ratios=[10, 1, 1],
    )
    read = gridspec.GridSpecFromSubplotSpec(
        4, 1,
        subplot_spec=grid[1],
        height_ratios=[18, 1, 1, 1]
    )

    # Plot write head
    for i, (image, title) in enumerate((write_weights, alloc_gates, write_gates)):
        if title == 'Write weights':
            add_subplot(write[i], image, title, ticks=True)
        else:
            add_subplot(write[i], image, title, ticks=False)

    # Plot read head
    for i, (image, title) in enumerate((read_weights, forward_modes, backward_modes, content_modes)):
        if title == 'Read weights':
            add_subplot(read[i], image, title, ticks=True)
        else:
            im = add_subplot(read[i], image, title, ticks=False)
    plt.subplots_adjust(hspace=0.4)

    # Colorbar
    cb_ax = fig.add_axes([0.92, 0.11, 0.02, 0.78])
    fig.colorbar(im, cax=cb_ax)
    plt.suptitle('DNC memory management')

    return fig2img(fig)


def ntm_img(info):
    def add_subplot(subplot, image, title, ticks):
        plt.subplot(subplot)
        im = plt.imshow(image, aspect='auto')
        plt.title(title, loc='left')
        plt.axis('on')
        if not ticks:
            plt.xticks([])
            plt.yticks([])
        return im

    # Write info
    write_weights = np.array(info['write_head']['write_weights']).T, 'Write weights'
    write_content_gates = np.array(info['write_head']['gates']).reshape(1, -1), 'Content gates'
    write_shifts = np.array(info['write_head']['shifts']).T, 'Shifts'

    # Read info
    read_weights = np.array(info['read_head']['read_weights']).T, 'Read weights'
    read_content_gates = np.array(info['read_head']['gates']).reshape(1, -1), 'Content gates'
    read_shifts = np.array(info['read_head']['shifts']).T, 'Shifts'

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    plt.gray()
    grid = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)
    write = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=grid[0],
        height_ratios=[12, 1, 2],
    )
    read = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=grid[1],
        height_ratios=[12, 1, 2]
    )

    # Plot write head
    for i, (image, title) in enumerate((write_weights, write_content_gates, write_shifts)):
        if title == 'Write weights':
            add_subplot(write[i], image, title, ticks=True)
        else:
            add_subplot(write[i], image, title, ticks=False)
    plt.subplots_adjust(hspace=0.4)

    # Plot read head
    for i, (image, title) in enumerate((read_weights, read_content_gates, read_shifts)):
        if title == 'Read weights':
            add_subplot(read[i], image, title, ticks=True)
        else:
            im = add_subplot(read[i], image, title, ticks=False)
    plt.subplots_adjust(hspace=0.4)

    # Colorbar
    cb_ax = fig.add_axes([0.92, 0.11, 0.02, 0.78])
    fig.colorbar(im, cax=cb_ax)
    plt.suptitle('NTM memory management')

    return fig2img(fig)
