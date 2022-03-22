"""EcgNet-HHA Dataset."""

import pathlib
import os
import collections

import numpy as np
import scipy
import scipy.signal
import skimage.draw
import matplotlib.pyplot as plt
import torch.utils.data
import hha
import pdb
import copy

class Ecg(torch.utils.data.Dataset):
    """EcgNet-HHA Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `ecgnet.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_type (string or list, optional): Type of target to use,
            ``Group'', 
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of ecg signal
                ``Group'' (int): group that ecg signal belongs to
            Defaults to ``Group''.
        mean (int, float, or np.array shape=(12,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the ECG signal. Defaults to 0 (signal is not shifted).
        std (int, float, or np.array shape=(12,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the ECG signal. Defaults to 1 (signal is not scaled).
        Hz (int): sampling rate of input ECG signals. Defaults to 500 samples per second.
        downsample (bool): whether or not to downsample sample, if True downsamples to half of sampling rates (Hz).
            Defaults to False (no downsampling).
        baseline (bool):  whether or not to remove baseline wander from signal. Defaults to False (assumes signals were already processed)
        notch (bool): whether or not to remove notch noise from signal. Defaults to False (assumes signals were already processed)
        preprocessed (bool): whether or not to use processed signals. Defaults to True (assumes signals were already processed)
        dimension (int): returns a 1D or 2D version of the ECG signals. Defaults to 1D. 
        external_test_location (string): Path to videos to use for external testing.
        file_list (string): Name of file list to use. Defaults to ''FileList_hh.csv''
    """

    def __init__(self, 
                 root=None,
                 split="train", 
                 target_type="Group",
                 mean=0.,
                 std=1.,
                 Hz=500,
                 downsample=False, 
                 baseline=False,
                 notch=False,
                 preprocessed=True,
                 dimension=1,
                 external_test_location=None,
                 file_list="FileList_hh.csv"
                 ):

        if root is None:
            root = hha.config.DATA_DIR

        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.Hz = Hz
        self.downsample = downsample
        self.baseline = baseline
        self.notch = notch
        self.preprocessed = preprocessed
        self.dimension = dimension
        self.external_test_location = external_test_location
        self.file_list = file_list
        self.fnames, self.outcome = [], []

        if split == "external":
            self.fnames = sorted(os.listdir(self.external_test_location))

        else:
            with open(self.folder / 'filelists' / self.file_list) as f:
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("FileName")
                splitIndex = self.header.index("Split")
                for i, line in enumerate(f):
                    lineSplit = line.strip().split(',')
                    fileName = lineSplit[filenameIndex]
                    fileMode = lineSplit[splitIndex].lower()
                    if self.preprocessed:
                        if split in ["all", fileMode] and (os.path.exists(self.folder / "Signals_preprocessed" / fileName)):
                            self.fnames.append(fileName)
                            self.outcome.append(lineSplit)
                    elif split in ["all", fileMode] and (os.path.exists(self.folder / "Signals" / fileName)):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)



    def __getitem__(self, index):
        # to support the indexing such that dataset[i] can be used to get ith sample
        # Find filename of ECG signals
        if self.split == "external":
            signal = os.path.join(self.external_test_location, self.fnames[index])

        elif self.preprocessed:
            signal = os.path.join(self.folder, "Signals_preprocessed", self.fnames[index])

        else:
            signal = os.path.join(self.folder, "Signals", self.fnames[index])

        # Load ECG signal into np.array
        signal = hha.utils.loadecg(signal, self.preprocessed).astype(np.float32)
        
        if self.notch:
            signal =_notch(signal, self.Hz)
        
        if self.baseline:
            signal =_baseline_wander_removal(signal, self.Hz)

        # Apply normalization
        if isinstance(self.mean, (float, int)):  
            signal = (signal.T - self.mean).T
        if isinstance(self.std, (float, int)):
            signal = (signal.T/self.std).T

        if self.dimension == 2:
            signal = np.expand_dims(signal, axis=0)

        if self.downsample:
            signal = _downsample_signal(signal, self.Hz, self.Hz/2)

        # Gather targets
        target = []
        for t in self.target_type:
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "Group":
                target.append(np.float32(self.outcome[index][self.header.index(t)])) 
            else:
                if self.split == "clinical_test" or self.split == "external_test":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))
        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
        
        signal = torch.Tensor(signal.astype(np.float32))
        return signal, target, self.fnames[index]


    def __len__(self):
        # len(dataset) returns the size of the dataset
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


def _notch(data, fs):
    # Pre-processing
    # 1. Moving averaging filter for power-line interference suppression:
    # averages samples in one period of the powerline
    # interference frequency with a first zero at this frequency.
    row,__ = data.shape
    #(data.shape)
    processed_data = np.zeros(data.shape)
    b = np.ones(int(0.02 * fs)) / 50.
    a = [1]
    for lead in range(0,row):
        X = scipy.signal.filtfilt(b, a, data[lead,:])
        processed_data[lead,:] = X
    return processed_data


def _baseline_wander_removal(data, sampling_frequency):
    row,__ = data.shape
    #print(data.shape)
    processed_data = np.zeros(data.shape)
    for lead in range(0,row):
        # Baseline estimation
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(data[lead,:], win_size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(baseline, win_size)
        # Removing baseline
        filt_data = data[lead,:] - baseline
        processed_data[lead,:] = filt_data
    return processed_data


def _downsample_signal(data, Hz, NewHz):
    row,__ = data.shape
    processed_data = np.zeros((row, 2500))
    for lead in range(0,row):
        ecgs = data[lead,:]
        ecg_ds = scipy.signal.resample(x=ecgs, num=2500, axis=0)
        processed_data[lead,:] = ecg_ds
    return processed_data

