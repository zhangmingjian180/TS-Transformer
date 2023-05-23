import numpy as np

from torch.utils.data import Dataset

from utils.utils import load_pickle
from utils.utils import normalize_DEAP

class MyDataset(Dataset):
    def __init__(self, paths, labels):
        super().__init__()
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        x = load_pickle(self.paths[index]).astype(np.float32)
        label = self.labels[index]
        return (x, label)


#$ base on a subject
# @param patrhs: [[], [], ...]
#
class MyDataset_SEED(Dataset):
    def __init__(self, paths, labels):
        super().__init__()
        all_paths = []
        all_labels = []
        
        for sub_list in paths:
            all_paths += sub_list
        for sub_list in labels:
            all_labels += sub_list
        
        self.paths = all_paths
        self.labels = all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x = load_pickle(self.paths[index]).astype(np.float32)
        label = self.labels[index]
        return (x, label)


class MyDataset_DEAP(Dataset):
    def __init__(self, x, y, sample_length, window_length):
        super().__init__()
        assert x.shape[2] % window_length == 0, "The length of window is error!"
        self.trial_num = x.shape[0]
        self.trial_length = x.shape[2]
        
        self.x = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
        self.y = y
        self.sample_length = sample_length
        self.window_length = window_length
        self.sample_of_trial = (self.trial_length - self.sample_length) // self.window_length + 1

    def __len__(self):
        return self.trial_num * self.sample_of_trial
    
    def __getitem__(self, index):
        x_start_point = index // self.sample_of_trial * self.trial_length + index % self.sample_of_trial * self.window_length
        return normalize_DEAP(self.x[:, x_start_point:x_start_point+self.sample_length]), self.y[index//self.sample_of_trial]


def gen_MyDataset_DEAP(sample_length, window_length):
    class MyDataset_DEAP(Dataset):
        def __init__(self, x, y):
            super().__init__()
            assert x.shape[2] % window_length == 0, "The length of window is error!"
            self.trial_num = x.shape[0]
            self.trial_length = x.shape[2]
        
            self.x = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
            self.y = y
            self.sample_length = sample_length
            self.window_length = window_length
            self.sample_of_trial = (self.trial_length - self.sample_length) // self.window_length + 1

        def __len__(self):
            return self.trial_num * self.sample_of_trial
    
        def __getitem__(self, index):
            x_start_point = index // self.sample_of_trial * self.trial_length + index % self.sample_of_trial * self.window_length
            return normalize_DEAP(self.x[:, x_start_point:x_start_point+self.sample_length]), self.y[index//self.sample_of_trial]
    
    return MyDataset_DEAP
