import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

# Dataset loader class for deer and diamond walk
# Arguments: root_dir - path to images
#            transforms - transformations to be applied
class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filelist = os.listdir(self.root_dir)
        self.filelist.sort()
        self.RGBtransform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        sample = Image.open(self.root_dir + '/' + self.filelist[idx])
        if self.transform:
            edge_sample = self.transform(sample)
        return (edge_sample, idx, self.RGBtransform(sample))

# Dataset loader class for citycenter and newcollege
# Arguments: root_dir - path to images
#            leftright - left or right camera pan
#            interval - interval for which to subset frames
#            transforms - transformations to be applied
# Member Function: getorig - obtain original RGB image
class realscenes(Dataset):
    def __init__(self, root_dir, leftright, interval, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.leftright = leftright
        self.interval = interval
        self.filelist = os.listdir(self.root_dir)
        self.filelist.sort()
        self.RGBtransform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.filelist) // (self.interval * 2)

    def __getitem__(self, idx):
        idx *= (self.interval * 2)
        if self.leftright == 'right':
            idx += 1
        sample = Image.open(self.root_dir + '/' + self.filelist[idx])
        if self.transform:
            edge_sample = self.transform(sample)
        return (edge_sample, idx)

    def getorig(self, idx):
        idx *= (self.interval * 2)
        if self.leftright == 'right':
            idx += 1
        sample = Image.open(self.root_dir + '/' + self.filelist[idx])
        return self.RGBtransform(sample)

# Dataset loader class for SCAMP dataset
# Arguments: root_dir - path to images
#            transforms - transformations to be applied
class scamp(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filelist = os.listdir(self.root_dir)
        self.filelist.sort()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        sample = Image.open(self.root_dir + '/' + self.filelist[idx])
        if self.transform:
            edge_sample = self.transform(sample)
        return (edge_sample, idx)