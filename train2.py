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
from dataset import MyCustomDataset2
from efficientnet_pytorch import EfficientNet


def train():
    epochs = 2000
    iteration = 10

    X_train, X_val, Y_train, Y_val = load_csv('./data/train.csv')
    train_dataset = MyCustomDataset2(X_data=X_train, Y_data=Y_train, mode='train')
    validation_dataset = MyCustomDataset2(X_data=X_val, Y_data=Y_val, mode='validation')

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8)
    valid_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = EfficientNet.from_pretrained('efficientnet-b7')
    model._fc = nn.Linear(in_features=2560, out_features=50, bias=True)
    model._swish = nn.Softmax()
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(r'D:\dacon_artist\saved_model\num2\best_0.5131022930145264_E_400.pth'))
    print(model)

    # print(model)
    best_model_weights = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
            # print('outputs:', outputs.shape)
            _, preds = torch.max(outputs, 1)
            # print(preds)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)
            train_running_corrects += torch.sum(preds == labels.data)

        train_epoch_loss = train_running_loss / len(train_dataset)
        train_epoch_acc = train_running_corrects / len(train_dataset)

        print('Training Loss: {:.4f}, Acc: {:.4f}' .format(train_epoch_loss, train_epoch_acc))

        scheduler.step()

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

            save_path_1 = r'D:\dacon_artist\saved_model\num2\best_{}_E_{}.pth'
            save_path_2 = r'D:\dacon_artist\saved_model\num2\iter_{}_E_{}.pth'
            if valid_epoch_acc > best_acc:
                print('********** Saving best model **********')
                best_acc = valid_epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path_1.format(best_acc, epoch))
            if epoch % 100 == 0:
                print('********** Saving best model **********')
                torch.save(model.state_dict(), save_path_2.format(valid_epoch_loss, epoch))

        print(time.time() - now)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('GPU is', device)
    train()