import torch
from torch.utils.data import Dataset
import os
import numpy as np

class ECGDataset(Dataset):
     
    def __init__(self, dataset):
        super(ECGDataset, self).__init__()

        # Load samples
        samples = dataset["samples"]

        # Convert to torch tensor
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)

        # Load labels
        labels = dataset.get("labels")
        if labels is not None and isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        self.samples = samples.float()
        self.labels = labels.long() 

        self.len = samples.shape[0]

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        sample = {
            'samples': self.samples[index],
            'labels': int(self.labels[index])
        }

        return sample

    def __len__(self):
        return self.len
