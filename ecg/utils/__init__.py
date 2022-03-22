"""Utility functions for videos, plotting and computing performance metrics."""

import os
import typing

#import cv2  # pytype: disable=attribute-error
import matplotlib
import numpy as np
import torch
import tqdm
import pdb
import sklearn
from sklearn.metrics import f1_score, classification_report, confusion_matrix, average_precision_score, roc_auc_score

from . import signal_dev
from . import signal_labels
from . import signal_auprc


def loadecg(filename: str, preprocessed:bool) -> np.ndarray:
    """Loads a ecg signal from a file.

    Args:
        filename (str): filename of ecg signal
        preprocessed (bool): True or False for preprocessed signals
    Returns:
        A np.ndarray with dimensions  [sequence_length, channels]. The
        values will be float16's.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    ecg_signal = np.load(filename, mmap_mode='r')
    if preprocessed is False:
        ecg_signal = ecg_signal.T
    
    channels, sequence_length = ecg_signal.shape
    assert (channels == 12), "Channels are not set to 12"
    if sequence_length > 5000:
        ecg_signal = ecg_signal[:,:5000]
        channels, sequence_length = ecg_signal.shape
    if not channels == 12:
        raise NameError('Channel length is not 12')
    if not sequence_length == 5000:
        raise NameError('Length of signal must be 5000')
    return ecg_signal

def loadecg_signals(filename: str, preprocessed:bool) -> np.ndarray:
    """Loads a ecg signal from a file.

    Args:
        filename (str): filename of ecg signal
        preprocessed (bool): True or False for preprocessed signals
    Returns:
        A np.ndarray with dimensions  [sequence_length, channels]. The
        values will be float16's.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    ecg_signal = np.load(filename, mmap_mode='r')
    if preprocessed is False:
        ecg_signal = ecg_signal.T
    
    channels, sequence_length = ecg_signal.shape
    assert (channels == 12), "Channels are not set to 12"
    if sequence_length > 5000:
        ecg_signal = ecg_signal[:,:5000]
        channels, sequence_length = ecg_signal.shape
    if not channels == 12:
        raise NameError('Channel length is not 12')
    if not sequence_length == 5000:
        raise NameError('Length of signal must be 5000')
    return ecg_signal


def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expsignalected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """
    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    x_arr = None
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.view(x.shape[1], -1)
        if x_arr is None:
            x_arr = x.numpy().T
        else:
            x_arr
            x_arr = np.concatenate([x_arr, x.numpy().T], axis=0)
    mean = np.mean(x_arr, axis=0)
    std = np.std(x_arr, axis=0)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    return mean,std



def plot_signal(data):
    ''' checking for data inputs are still okay
    '''
    k = 0
    columns = 1
    rows = 12
    lead_order = ['1', '2', '3', 'R', 'L', 'F', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, ax_array = matplotlib.pyplot.subplots(rows, columns,squeeze=False, figsize=(15,20))
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            signal1 = data[i,:]
            x1 = np.linspace(0, 10, signal1.shape[0])
            y1 = np.squeeze(signal1)
            axes.plot(x1, y1, 'k-')
            axes.set_xlim(x1.min(), x1.max())
            axes.set_ylim(y1.min(), y1.max())
            axes.set_title('Lead ' + lead_order[i], size=20)
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
    matplotlib.pyplot.subplots_adjust(hspace=0.9)
    matplotlib.pyplot.suptitle('Subject 1,\n Raw ECG', size=20)
    matplotlib.pyplot.savefig('/home/users/jntorres/jessica/stanford_ECG_projects/ecgnet_hha/test/subject_1.png', bbox_inches='tight', transparent=False,  format='png', dpi=500)
    #plt.show()


def bootstrap(a, b, func, samples=10000):
    """Computes a bootstrapped confidence intervals for ``func(a, b)''.

    Args:
        a (array_like): first argument to `func`.
        b (array_like): second argument to `func`.
        func (callable): Function to compute confidence intervals for.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int, optional): Number of samples to compute.
            Defaults to 10000.

    Returns:
       A tuple of (`func(a, b)`, estimated 5-th percentile, estimated 95-th percentile).
    """
    a = np.array(a)
    b = np.array(b)

    bootstraps = []
    for _ in range(samples):
        ind = np.random.choice(len(a), len(a))
        bootstraps.append(func(a[ind], b[ind]))
    bootstraps = sorted(bootstraps)

    return func(a, b), bootstraps[round(0.05 * len(bootstraps))], bootstraps[round(0.95 * len(bootstraps))]


def latexify():
    """Sets matplotlib params to appear more like LaTeX.

    Based on https://nipunbatra.github.io/blog/2014/latexify.html
    """
    params = {'backend': 'pdf',
              'axes.titlesize': 8,
              'axes.labelsize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
              }
    matplotlib.rcParams.update(params)


def dice_similarity_coefficient(inter, union):
    """Computes the dice similarity coefficient.

    Args:
        inter (iterable): iterable of the intersections
        union (iterable): iterable of the unions
    """
    return 2 * sum(inter) / (sum(union) + sum(inter))


__all__ = ["signal", "loadecg", "get_mean_and_std", "bootstrap", "latexify", "dice_similarity_coefficient"]
