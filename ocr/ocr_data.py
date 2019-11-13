import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return np.asarray(img)

def np_img_to_tensor(np_img):
    return torch.from_numpy(np_img.transpose((2, 0, 1)))

class WordsDataset(torch.utils.data.Dataset):
    """
    Creates dataset for images of words.
    """

    def __init__(self, csv_file, data_dir):
        """
        csv_file: CSV file with image file names in first column, and label in
            second column.
        data_dir: Directory to find image files.
        """
        self.words_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.words_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.words_frame.iloc[idx, 0])
        img = np_img_to_tensor(load_image(img_path))

        label = self.words_frame.iloc[idx, 1]

        return img, label

if __name__ == '__main__':
    csv_file = sys.argv[1]
    data_dir = sys.argv[2]

    data = WordsDataset(csv_file, data_dir)
    loader = torch.utils.data.DataLoader(data, batch_size=64)
    for sample in loader:
        print(sample)
        break