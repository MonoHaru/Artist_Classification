import os, sys
import torch
import torch.nn as nn
# import torchsummary
import copy
from torch.utils.data import DataLoader
from utils.csv import load_csv
from dataset import MyTestDataset
from efficientnet_pytorch import EfficientNet
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import time
from torch.utils.data import DataLoader
from utils.csv import load_csv
from efficientnet_pytorch import EfficientNet



def valid():
    epochs = 2000
    iteration = 10

    sample = pd.read_csv(r'./data/sample_submission.csv')
    X_train = load_csv('./data/test.csv', mode='test')
    test_dataset = MyTestDataset(X_train)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)


    model = EfficientNet.from_name('efficientnet-b7', num_classes=50)
    model._fc = nn.Linear(in_features=2560, out_features=50, bias=True)
    model._swish = nn.Softmax()
    model = torch.nn.DataParallel(model).to(device)
    # model = model.to(device)
    # print(model)
    # model.load_state_dict(torch.load(r'D:\dacon_artist\saved_model\num2\best_0.42349958419799805_E_200.pth'))
    # model = model.to(device)
    print(model)

    # print(model)
    best_model_weights = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=0)
    model.load_state_dict(torch.load(r'D:\dacon_artist\saved_model\num2\best_0.5131022930145264_E_400.pth'))
    # model = torch.load(r'D:\dacon_artist\saved_model\num2\best_0.12933219969272614_E_0.pth')
    i = 0

    model.eval()
    with torch.no_grad():
        for test_inputs in test_dataloader:
            test_inputs = test_inputs.to(device)
            outputs = model(test_inputs)
            _, preds = torch.max(outputs, 1)
            artist = test_dataset.get_name(preds)
            print(preds)
            print(artist)
            sample['artist'][i] = artist
            i += 1

    sample.to_csv(r'./data/submit.csv', index=False)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('GPU is', device)
    valid()