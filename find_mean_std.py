import os, sys
import torch
from torch.utils.data import DataLoader
from utils.csv import load_csv
from dataset import MyCustomDataset2
import numpy as np

def go():
    epochs = 2000
    iteration = 10

    X_train, X_val, Y_train, Y_val = load_csv('./data/train.csv')
    train_dataset = MyCustomDataset2(X_data=X_train, Y_data=Y_train, mode='train')
    validation_dataset = MyCustomDataset2(X_data=X_val, Y_data=Y_val, mode='validation')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=2)

    print_stats(train_dataloader)


def print_stats(dataset):
    imgs = np.array([img.numpy() for img, _ in dataset])
    print(f'shape: {imgs.shape}')

    min_r = np.min(imgs, axis=(2, 3))[:, 0].min()
    min_g = np.min(imgs, axis=(2, 3))[:, 1].min()
    min_b = np.min(imgs, axis=(2, 3))[:, 2].min()

    max_r = np.max(imgs, axis=(2, 3))[:, 0].max()
    max_g = np.max(imgs, axis=(2, 3))[:, 1].max()
    max_b = np.max(imgs, axis=(2, 3))[:, 2].max()

    mean_r = np.mean(imgs, axis=(2, 3))[:, 0].mean()
    mean_g = np.mean(imgs, axis=(2, 3))[:, 1].mean()
    mean_b = np.mean(imgs, axis=(2, 3))[:, 2].mean()

    std_r = np.std(imgs, axis=(2, 3))[:, 0].std()
    std_g = np.std(imgs, axis=(2, 3))[:, 1].std()
    std_b = np.std(imgs, axis=(2, 3))[:, 2].std()

    print(f'min: {min_r, min_g, min_b}')
    print(f'max: {max_r, max_g, max_b}')
    print(f'mean: {mean_r, mean_g, mean_b}')
    print(f'std: {std_r, std_g, std_b}')

#
# import numpy as np
#
# meanRGB = [np.mean(x.numpy(), axis=(2,3)) for x,_ in train_dataset]
# stdRGB = [np.std(x.numpy(), axis=(2,3)) for x,_ in train_dataset]
#
# meanR = np.mean([m[0] for m in meanRGB])
# meanG = np.mean([m[1] for m in meanRGB])
# meanB = np.mean([m[2] for m in meanRGB])
#
# stdR = np.mean([s[0] for s in stdRGB])
# stdG = np.mean([s[1] for s in stdRGB])
# stdB = np.mean([s[2] for s in stdRGB])
#
# print(meanR, meanG, meanB)
# print(stdR, stdG, stdB)

if __name__ == '__main__':
    # freeze_support()
    go()