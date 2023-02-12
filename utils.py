
import os
from skimage import io, transform
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ImagesDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = str(self.labels.iloc[idx, 0])
        img_path = os.path.join(self.root_dir, img_name + '.jpg')
        image = io.imread(img_path)
        
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = (image, label, img_name)

        return sample
