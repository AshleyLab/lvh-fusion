"""LVH-Fusion Dataset."""

import pathlib
import os
import collections

import numpy as np
import numpy.ma as ma
import pandas as pd
import skimage.draw
import torch.utils.data
import hha
import pdb
import cv2

class Echo(torch.utils.data.Dataset):
    """LVH-Fusion Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
        file_list="FileList_hh.csv",
        singleframe=False,
        seed=2021,
        singleframe_ed=False)
        segmentation_mask=False,
        segmentation_mask_invert=False,
        downsample=None)
    """

    def __init__(self, root=None,
                 split="train", 
                 target_type="Group",
                 mean=0., 
                 std=1.,
                 length=16, 
                 period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 file_list="FileList_hh.csv",
                 singleframe=False,
                 seed=2021,
                 singleframe_ed=False,
                 segmentation_mask=False,
                 segmentation_mask_invert=False,
                 downsample=None,
                 segmentation_params=None,
                 device=None,
                 segmentation_outline=False):

        if root is None:
            root = hha.config.DATA_DIR

        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.file_list = file_list
        self.fnames, self.outcome = [], []
        self.singleframe = singleframe
        self.seed = seed
        self.singleframe_ed = singleframe_ed
        self.segmentation_mask = segmentation_mask
        self.segmentation_mask_invert = segmentation_mask_invert
        self.downsample = downsample
        self.segmentation_params = segmentation_params
        self.device = device
        self.segmentation_outline = segmentation_outline

        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
            print(self.fnames)
        else:
            with open(self.folder / 'filelists' /self.file_list) as f:
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("FileName")
                splitIndex = self.header.index("Split")
                for line in f:
                    lineSplit = line.strip().split(',')
                    fileName = lineSplit[filenameIndex]
                    fileMode = lineSplit[splitIndex].lower()

                    if split in ["all", fileMode] and os.path.exists(self.folder / "Videos" / fileName):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])



    def __getitem__(self, index):
        # to support the indexing such that dataset[i] can be used to get ith sample
        # Find filename of video
        if self.split == "external_test":
            video = os.path.join(self.external_test_location, self.fnames[index])
        else:
            video = os.path.join(self.folder, "Videos", self.fnames[index])
        
        ## Downsample data and than upscaling for match 112x112 dims.
        if self.downsample is not None:
            video = hha.utils.loadvideo_downsample(video, self.downsample).astype(np.float32)
        else:
            # Load video into np.array
            video = hha.utils.loadvideo(video).astype(np.float32)            

        ## Single frame, random
        if self.singleframe:
            c,fr,h,w = video.shape
            np.random.seed(self.seed)
            selectedframe = np.random.choice(fr, 1)
            video = np.tile(video[:,selectedframe,:,:], (1,fr,1,1))

        ## Single frame, end-diastolic
        if self.singleframe_ed:
            c,fr,h,w = video.shape
            sf_lde = pd.read_csv(self.folder / 'filelists/hha_largest_size_est.csv')
            sf_lde_dict = dict(zip(sf_lde.Filename, sf_lde.Frame))
            selectedframe = int(sf_lde_dict[self.fnames[index]])
            video = np.tile(np.expand_dims(video[:,selectedframe,:,:], axis=1), (1,fr,1,1))

        ## Masking out segmentation only ventricle
        if self.segmentation_mask:
            path_to_labels='labels'
            labels = np.load(path_to_labels + self.fnames[index].split(".")[0] + '.npy').astype(float)
            mask = ma.masked_where(labels<=0, labels)
            #print("Mask shape", mask.mask.shape)
            video[0,mask.mask]=0.0
            video[1,mask.mask]=0.0
            video[2,mask.mask]=0.0

        ## Masking out segmentation, all but ventricle
        if self.segmentation_mask_invert:
            path_to_labels='hha-video-dir/Videos/labels'
            labels = np.load(path_to_labels + self.fnames[index].split(".")[0] + '.npy').astype(float)
            mask = ma.masked_where(labels>0, labels)
            video[0,mask.mask]=0.0
            video[1,mask.mask]=0.0
            video[2,mask.mask]=0.0

        if self.segmentation_params is not None:  ###############################
            path_to_labels='hha-video-dir/Videos/labels'
            labels = np.load(path_to_labels + self.fnames[index].split(".")[0] + '.npy')
            segmentation_map = (labels > 0).astype(np.float32)
            segmentation_map = np.expand_dims(segmentation_map, 0)
            segmentation_map =  torch.Tensor(segmentation_map)

            #segmentation_map = torch.Tensor(np.repeat(segmentation_map, 3, 0))
            video = torch.Tensor(video)
            print("Segmentation map:")
            print(segmentation_map.shape)
            expander = SegmentationExpander(self.segmentation_params["expand"], self.device)
            if self.segmentation_params['mask']:
                segmentation_map = expander.expand(segmentation_map)
                if self.segmentation_params['reverse']:
                    segmentation_map = ~segmentation_map
                video = video * segmentation_map.unsqueeze(1)
            elif self.segmentation_params['rect']:
                segmentation_map = expander.expand_rectangle(segmentation_map)
                if self.segmentation_params['reverse']:
                    segmentation_map = ~segmentation_map
                video = video * segmentation_map.unsqueeze(1)
            elif self.segmentation_params['mitral']:
                mitral_mask = expander.mitral_cover(segmentation_map)
                if self.segmentation_params['reverse']:
                    mitral_mask = ~mitral_mask
                video = video * (~mitral_mask).unsqueeze(1)
            video = np.squeeze(video).detach().numpy()
            #print(video.shape)
            v_out = video.transpose(( 1, 2, 3, 0))
            capture = cv2.VideoCapture('hha-video-dir/Videos/' + self.fnames[index])
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            #print(fps)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            cropSize = (112,112)
            newFile_ext = '_westSeg'
            fileName = self.fnames[index].split(".")[0] + newFile_ext 
            destinationFolder = 'echonet_hha/hha-video-dir/modified_vids/'
            video_filename = destinationFolder + fileName + '.avi'
            out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)
            for i in range(v_out.shape[0]):
                finaloutput = v_out[i,:,:,:]
                out.write(np.uint8(finaloutput))
            out.release()
            ###############################

        if self.segmentation_outline:
            path_to_labels='echonet/dynamic-master/hha_out/labels/'
            labels = np.load(path_to_labels + self.fnames[index].split(".")[0] + '.npy')
            seg = (labels > 0).astype(np.float32)
            video = np.expand_dims(seg, 0)
            video = np.repeat(video, 3, 0)


        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            if 'test' in self.split:
                #print("setting seed properly")
                np.random.seed(self.seed)
                start = np.random.choice(f - (length - 1) * self.period, self.clips)
            else:
                start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            #key = os.path.splitext(self.fnames[index])[0]
            #print(key)
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
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return video, target, self.fnames[index]


    def __len__(self):
        # len(dataset) returns the size of the dataset
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class SegmentationExpander():
    """Helper class for masking ablations. 
    """
    def __init__(self, radius, device):
        conv = np.zeros((1, 1, 2*radius + 1, 2*radius + 1)).astype(np.float32)
        for i in range(len(conv[0,0])):
            for j in range(len(conv[0,0,0])):
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    conv[0, :, i, j] = 1
        self.conv = torch.from_numpy(conv).to(device)
        self.radius = radius
        self.device = device

    def expand(self, mask):
        # Expands mask by self.radius
        shape = mask.shape
        mask = mask.reshape(-1, 1, *shape[2:])
        mask = torch.nn.functional.conv2d(mask, self.conv, padding=self.radius) > 0
        return mask.reshape(shape)

    def expand_rectangle(self, mask):
        # Finds rectangle around mask and then expands it
        # Union of masks for all frames of each video
        shape = mask.shape
        print("shape:")
        print(shape)
        comp = (mask > 0).sum(1) > 0
        # Outer product of binary mask for h and binary mask for w
        comp_rect = torch.bmm((comp.sum(1).unsqueeze(2) > 0).float(),
                              (comp.sum(2).unsqueeze(1) > 0).float())
        mask = comp_rect.unsqueeze(1).expand(mask.shape[0], 32, 112, 112).transpose(3,2)
        mask = self.expand(mask)
        return mask

    def mitral_cover(self, mask):
        # Finds mask approximately covering mitral valve
        sums = mask.sum(3).reshape(-1, 112).cpu().numpy()
        nonz = [np.nonzero(s)[0] for s in sums]
        upper = [int(self.min(n) / 3 + 2 * self.max(n) / 3) for n in nonz]
        masks = [np.zeros((112, 112)) for _ in upper]
        for i in range(len(upper)):
            masks[i][upper[i]:] = 1
        masks = np.array(masks).reshape(mask.shape)
        masks = torch.logical_and(torch.from_numpy(masks).to(self.device), self.expand(mask))
        return masks

    def min(self, n):
        try:
            return min(n)
        except:
            return 111

    def max(self, n):
        try:
            return max(n)
        except:
            return 111
