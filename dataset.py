import os
from skimage import io, img_as_float32, img_as_int
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob

def read_video(name, frame_shape):
    """
    Read video which can be:
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array([img_as_float32(io.imread(os.path.join(name, str(frames[idx], encoding="utf-8")))) for idx in range(num_frames)])
    elif name.lower().endswith(".gif") or name.lower().endswith(".mp4"):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(
        self,
        root_dir="/home/lh/repo/datasets/face-video-preprocessing/vox-png/",
        frame_shape=(256, 256, 3),
        id_sampling=True,
        is_train=True,
        random_seed=0,
        pairs_list=None,
        augmentation_params={
            "rotation_param": {"degrees": 35},
            # "perspective_param":{"pers_num": 30, "enlarge_num": 40},
            # "flip_param": {"horizontal_flip": False, "time_flip": False},
            "scale_param": {"ratio": [0.7, 1.2]},
            "translate_param": {"x_ratio": 1/16, "y_ratio": 1/16}, 
            "jitter_param": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        },
    ):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        # import pdb; pdb.set_trace()
        if os.path.exists(os.path.join(root_dir, "train")):
            assert os.path.exists(os.path.join(root_dir, "test"))
            if id_sampling:
                train_videos = {os.path.basename(video).split("#")[0] for video in os.listdir(os.path.join(root_dir, "train"))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, "train"))
            test_videos = os.listdir(os.path.join(root_dir, "test"))
            self.root_dir = os.path.join(self.root_dir, "train" if is_train else "test")
        else:
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + "*.mp4")))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path+"/img")
            num_frames = len(frames)
            try:
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
                video_array = [img_as_float32(io.imread(os.path.join(path, str(frames[idx], encoding="utf-8")))) for idx in frame_idx]
            except:
                print(path, frame_idx)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        # if self.transform is not None:
        #     video_array = self.transform(video_array)

        if self.is_train:
            source = np.array(video_array[0], dtype="float32")
            driving = np.array(video_array[1], dtype="float32")

            driving = driving.transpose((2, 0, 1))
            source = source.transpose((2, 0, 1))
            if self.transform is not None:
                # video_array = self.transform(video_array)
                source_aug = np.array(self.transform([video_array[0]])[0])
                driving_aug = np.array(self.transform([video_array[1]])[0])
                source_aug = source_aug.transpose((2, 0, 1))
                driving_aug = driving_aug.transpose((2, 0, 1))
            else:
                source_aug, driving_aug = None, None
            return source, driving, source_aug, driving_aug
            # return source, driving
        else:
            video = np.array(video_array, dtype="float32")
            video = video.transpose((3, 0, 1, 2))
        
            return video

class AudioDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, 
        root_dir, 
        frame_shape=(256, 256, 3), 
        id_sampling=False, 
        is_train=True,
        is_audio=False,
        random_seed=0, 
        augmentation_params={
            "rotation_param": {"degrees": 35},
            # "perspective_param":{"pers_num": 30, "enlarge_num": 40},
            # "flip_param": {"horizontal_flip": False, "time_flip": False},
            "scale_param": {"ratio": [0.7, 1.2]},
            "translate_param": {"x_ratio": 1/16, "y_ratio": 1/16}, 
            "jitter_param": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
        },
    ):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir,'mfcc')
        self.image_dir = os.path.join(root_dir,'Image')
        self.pose_dir = os.path.join(root_dir,'pose')

        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling

        print("Use predefined train-test split.")
        train_videos =  glob.glob(os. path.join(self.image_dir,"*/train/*.mp4"))
        test_videos =  glob.glob(os. path.join(self.image_dir,"*/test/*.mp4"))

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train
        self.is_audio = is_audio

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx].split('.')[0]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            dir_name = self.videos[idx].split(os.sep)[-3]
        
        if self.is_train:
            mode = 'train'
        else:
            mode = 'test'


        video_name = self.videos[idx]
        video_array = read_video(video_name, self.frame_shape)

        num_frames = len(video_array)
        frame_idx = np.sort(np.random.choice(range(3, num_frames-3), replace=True, size=2))
        # video_array = video_array[frame_idx]

        basename = os.path.basename(video_name).split('.')[0]
        # mfcc loading
        # r = np.random.choice([x for x in range(3, 8)])
        # num_frames = len(video_array)

        # example_image = img_as_float32(io.imread(os.path.join(path, str(r)+'.png')))
        # example_image = video_array[r]
        
        if self.is_audio:
            audio_path = os.path.join(self.audio_dir, dir_name, f'{mode}_mfcc')
            mfccs = np.load(os.path.join(audio_path, basename +'.npy'))[frame_idx, 1:]
        
        # pose_path = os.path.join(self.pose_dir, dir_name, f'{mode}_pose')
        # kpts_path = os.path.join(self.audio_dir, name, f'{mode}_2d_sparse')

        # path = os.path.join(self.image_dir, name)
        
        # poses = np.load(os.path.join(pose_path, basename +'.npy')) [frame_idx, :-1]
        video_array = np.array(video_array)[frame_idx]
     
        # driving = np.array(video_array, dtype='float32')
        # spatial_size = np.array(driving.shape[1:3][::-1])[np.newaxis]
        # driving_pose = np.array(poses, dtype='float32')
        # example_image = np.array(example_image, dtype='float32')

        if self.is_train:
            source = np.array(video_array[0], dtype="float32")
            driving = np.array(video_array[1], dtype="float32")

            driving = driving.transpose((2, 0, 1))
            source = source.transpose((2, 0, 1))
            if self.transform is not None:
                # video_array = self.transform(video_array)
                source_aug = np.array(self.transform([video_array[0]])[0])
                driving_aug = np.array(self.transform([video_array[1]])[0])
                source_aug = source_aug.transpose((2, 0, 1))
                driving_aug = driving_aug.transpose((2, 0, 1))
            else:
                source_aug, driving_aug = None, None
            
            if self.is_audio:
                source_mfcc = np.array(mfccs[0], dtype='float32')
                driving_mfcc = np.array(mfccs[1], dtype='float32')
                return source, driving, source_mfcc, driving_mfcc
            else:
                return source, driving, source_aug, driving_aug
            # return source, driving
        else:
            video = np.array(video_array, dtype="float32")
            # video = video.transpose((3, 0, 1, 2))
            video = video.transpose((0, 3, 1, 2))

        return video

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=75):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs["source"].isin(videos), pairs["driving"].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index[pairs["driving"].iloc[ind]], name_to_index[pairs["source"].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {"driving_" + key: value for key, value in first.items()}
        second = {"source_" + key: value for key, value in second.items()}

        return {**first, **second}
