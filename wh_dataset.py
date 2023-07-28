import os
from PIL import Image
from torch.utils.data import Dataset
import random
import torch

class WHDataset(Dataset):
    def __init__(self, txt_file, base_path, transform=None, shuffle=False):
        self.data_list = []
        with open(txt_file, 'r') as f:
            for line in f:
                paths = line.strip().split()
                image_path = os.path.join(base_path, paths[0])
                mask_path = os.path.join(base_path, paths[1])
                self.data_list.append((image_path, mask_path))
        self.transform = transform
        self.shuffle = shuffle
        self.mask_values = []
        if self.shuffle:
            random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, mask_path = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            #mask = mask.unsqueeze(0)             
            #mask = mask.to(memory_format=torch.channels_last)
        batch = {'image':image,'mask':mask}
        return batch
