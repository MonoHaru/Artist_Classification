import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
# import torchsummary
import copy
import time
from torch.utils.data import DataLoader
from utils.csv import load_csv
from dataset import MyCustomDataset
from efficientnet_pytorch import EfficientNet


def valid():

    X_train, X_val, Y_train, Y_val = load_csv('./data/train.csv')
    validation_dataset = MyCustomDataset(X_data=X_val, Y_data=Y_val, mode='validation')
    validation_dataset2 = MyCustomDataset(X_data=X_val, Y_data=Y_val, mode='validation', mode2=True)

    valid_dataloader = DataLoader(validation_dataset, batch_size=40, shuffle=False, num_workers=6)
    valid_dataloader2 = DataLoader(validation_dataset2, batch_size=40, shuffle=False, num_workers=6)

    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=50)
    model._fc = nn.Linear(in_features=1280, out_features=50, bias=True)
    model._swish = nn.Softmax()
    model.load_state_dict(torch.load(r'D:\dacon_artist\saved_model\num1\best_0.5393068194389343_E_1030.pth'))
    model = model.to(device)

    # print(model)
    best_model_weights = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()

    valid_running_loss = 0.0
    valid_running_corrects = 0.0


    model.eval()
    with torch.no_grad():
        for val_inputs, val_labels in valid_dataloader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_inputs)
            _, preds = torch.max(outputs, 1)
            # print(preds)
            loss = criterion(outputs, val_labels)

            valid_running_loss += loss.item() * val_inputs.size(0)
            valid_running_corrects += torch.sum(preds == val_labels.data)

        valid_epoch_loss = valid_running_loss / len(validation_dataset)
        valid_epoch_acc = valid_running_corrects / len(validation_dataset)

        print('Validation1 Loss: {:.4f}, Acc: {:.4f}'.format(valid_epoch_loss, valid_epoch_acc))

    valid_running_loss = 0.0
    valid_running_corrects = 0.0

    model.eval()
    with torch.no_grad():
        for val_inputs, val_labels in valid_dataloader2:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, val_labels)

            valid_running_loss += loss.item() * val_inputs.size(0)
            valid_running_corrects += torch.sum(preds == val_labels.data)

        valid_epoch_loss = valid_running_loss / len(validation_dataset2)
        valid_epoch_acc = valid_running_corrects / len(validation_dataset2)

        print('Validation2 Loss: {:.4f}, Acc: {:.4f}'.format(valid_epoch_loss, valid_epoch_acc))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('GPU is', device)
    valid()