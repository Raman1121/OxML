import os
import re
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.transforms import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, 'img_' + str(self.labels_df.iloc[idx]['id']) + '.png')
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx]['malignant'] + 1

        if self.transform:
            image = self.transform(image)

        return image, label
