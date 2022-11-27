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
    model.load_state_dict(torch.load(r'D:\dacon_artist\saved_model\num1\best_0.16906170547008514_E_300.pth'))
    model = model.to(device)

    # print(model)
    best_model_weights = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), weight_decay=1e-5, lr=3e-2, momentum=0.9)
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

            save_path_1 = r'D:\dacon_artist\saved_model\num1\best_{}_E_{}.pth'
            save_path_2 = r'D:\dacon_artist\saved_model\num1\iter_{}_E_{}.pth'
            if valid_epoch_acc > best_acc:
                print('********** Saving best model **********')
                best_acc = valid_epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path_1.format(best_acc, epoch))
            if epoch % 1000 == 0:
                print('********** Saving iter model **********')
                torch.save(model.state_dict(), save_path_2.format(valid_epoch_loss, epoch))

        print(time.time() - now)

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('GPU is', device)
    train()