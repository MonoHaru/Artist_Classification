import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.csv import load_classes_csv
from sklearn import preprocessing

# define transform for training, validation
def transform(mode):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if mode == 'train':
        trans = transforms.Compose([
            # transforms.Resize(512),
            transforms.RandomRotation((-30,30)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        return trans
    elif mode == 'validation':
        trans = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        return trans

def transform2(mode):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if mode == 'train':
        trans = transforms.Compose([
            transforms.Resize((1200,1200)),
            # transforms.RandomPerspective(),
            transforms.RandomRotation((-30,30)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomCrop(600),
            # transforms.Resize((600, 600)),
            transforms.ToTensor(),
            # normalize
        ])
        return trans
    elif mode == 'validation':
        trans = transforms.Compose([
            transforms.Resize((600,600)),
            # transforms.CenterCrop(600),
            transforms.ToTensor(),
            # normalize
        ])
        return trans

def encoding_classes2index(labels):
    dicts = dict()
    for i, label in enumerate(labels):
        dicts[label] = i
    return dicts

class MyCustomDataset(Dataset):
    def __init__(self, X_data, Y_data, mode):
        self.X_data = X_data
        self.Y_data = Y_data
        self.mode = mode
        self.transform = transform(self.mode)
        self.classes = encoding_classes2index(load_classes_csv(root='./data/artists_info.csv'))
        self.one_hot_vector = F.one_hot(torch.arange(0, 50), num_classes=50)
        # print('self.classes:' ,self.classes)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = Image.open(self.X_data['img_path'][idx]).convert('RGB')
        label = self.classes[self.Y_data[idx]]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

class MyCustomDataset2(Dataset):
    def __init__(self, X_data, Y_data, mode):
        self.X_data = X_data
        self.Y_data = Y_data
        self.mode = mode
        self.transform = transform2(self.mode)
        self.classes = encoding_classes2index(load_classes_csv(root='./data/artists_info.csv'))
        self.one_hot_vector = F.one_hot(torch.arange(0, 50), num_classes=50)
        # print('self.classes:' ,self.classes)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = Image.open(self.X_data['img_path'][idx]).convert('RGB')
        label = self.classes[self.Y_data[idx]]
        if self.transform is not None:
            img = self.transform(img)
        if self.X_data['artist'][idx] != self.Y_data[idx]:
            print('좆됐어!!!! 둠황챠!!!')
        return img, label

class MyTestDataset(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        self.transform = transform('validation')
        self.classes = encoding_classes2index(load_classes_csv(root='./data/artists_info.csv'))
        print('self.classes: ', self.classes)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = Image.open(self.X_data['img_path'][idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_name(self, item):
        for key, value in self.classes.items():
            if item == value:
                return key
