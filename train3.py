import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchsummary
import copy
import time
import argparse
from torch.utils.data import DataLoader
from utils.csv import load_csv
from dataset import MyCustomDataset
from efficientnet_pytorch import EfficientNet


def train():
    epochs = 10000
    iteration = 100

    X_train, X_val, Y_train, Y_val = load_csv('./data/train.csv')
    train_dataset = MyCustomDataset(X_data=X_train, Y_data=Y_train, mode='train')
    validation_dataset = MyCustomDataset(X_data=X_val, Y_data=Y_val, mode='validation')

    train_dataloader = DataLoader(train_dataset, batch_size=148, shuffle=True, num_workers=3)
    valid_dataloader = DataLoader(validation_dataset, batch_size=148, shuffle=False, num_workers=6)

    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=50)
    model._fc = nn.Linear(in_features=1280, out_features=50, bias=True)
    model._swish = nn.Softmax()
    model = model.to(device)
    # print(model)
    best_model_weights = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), weight_decay=1e-5, lr=1e-1, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0)

    best_acc = 0

    for epoch in range(epochs):
        print('==========' * 10)
        print('Epch {}/{}' .format(epoch, epochs-1))
        print('==========' * 10)
        now = time.time()
        train_running_loss = 0.0
        train_running_corrects = 0.0
        valid_running_loss = 0.0
        valid_running_corrects = 0.0

        for inputs, labels in train_dataloader:
            model.train()
            # print('inputs:', inputs.shape)
            # print('labels:', labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            # print('outputs:', outputs)
            _, preds = torch.max(outputs, 1)
            # print(preds)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.item() * inputs.size(0)
            train_running_corrects += torch.sum(preds == labels.data)

        train_epoch_loss = train_running_loss / len(train_dataset)
        train_epoch_acc = train_running_corrects / len(train_dataset)

        print('Training Loss: {:.4f}, Acc: {:.4f}' .format(train_epoch_loss, train_epoch_acc))

        if epoch == 0 or epoch % iteration == 0:
            model.eval()
            with torch.no_grad():
                for val_inputs, val_labels in valid_dataloader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    outputs = model(val_inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, val_labels)

                    valid_running_loss += loss.item() * val_inputs.size(0)
                    valid_running_corrects += torch.sum(preds == val_labels.data)

                valid_epoch_loss = valid_running_loss / len(validation_dataset)
                valid_epoch_acc = valid_running_corrects / len(validation_dataset)

                print('Validation Loss: {:.4f}, Acc: {:.4f}'.format(valid_epoch_loss, valid_epoch_acc))

            save_path = r'D:\dacon_artist\saved_model\num1\best_{}_E_{}.pth'
            if valid_epoch_acc > best_acc:
                print('********** Saving best model **********')
                best_acc = valid_epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path.format(best_acc, epoch))

        print(time.time() - now)

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('GPU is', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', default='./data/train.csv', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', default='', required=True, help='path to validation dataset')
    parser.add_argument('--train_workers', default=3, type=int,help='number of training data loading workers')
    parser.add_argument('--valid_workers', default=6, type=int, help='number of validation data loading workers')
    parser.add_argument('--batch_size', default=148, type=int, help='input batch size')
    parser.add_argument('--valInterval', default=100, type=int, help='Intereval between each validation')
    parser.add_argument('--saved_model', default='', help='path to model to continue training')
    parser.add_argument('--optimizer', default='', help='optimizer which you choose')
    parser.add_argument('--lr', default=1, type=float, help='learning rate')
    parser.add_argument('--imgH', default=224, type=int, help='the height of the input image')
    parser.add_argument('--imgW', default=224, type=int, help='the width of the input image')
    parser.add_argument('--scheduler', default='', help='learning rate Scheduler')
    parser.add_argument('--epochs', default=10000, type=int, help='count of training about all training dataset')
    parser.add_argument('--Prediction', default='', type=str, help='Prediction stage(Loss function)')

    parser.add_argument('',default=, help=)

    opt = parser.parse_args()

    # if not opt.exp_name:
    #     opt.exp_name = f'{}'

    train()