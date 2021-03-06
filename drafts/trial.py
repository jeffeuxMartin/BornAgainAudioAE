import torch, torch.nn as nn
# for ii in range(400, 500):
ii = torch.randint(300, 600, (1,)).item()
if 1:
    xx = torch.randn(3, 1, ii, 128)
    # xx = x.cpu()
    print(xx.shape)
    print('\033[01;31m''After Conv''\033[0m')
    xx = nn.Conv2d(1, 12, (4, 1), stride=(2, 1), padding=(1, 0))(xx)
    xx = nn.MaxPool2d((2, 1), stride=(2, 1), padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)

    xx = nn.Conv2d(12, 24, (4, 2), stride=(2, 2), padding=(1, 0))(xx)
    xx = nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)

    xx = nn.Conv2d(24, 48, (4, 2), stride=(2, 2), padding=(1, 0))(xx)
    xx = nn.MaxPool2d((2, 2), stride=(2, 2), padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)

    xx = nn.Conv2d(48, 64, (4, 2), stride=(2, 2), padding=(1, 0))(xx)
    xx = nn.MaxPool2d((2, 2), stride=(1, 2), padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)
    
    u = xx.size(-2)
    xx = xx.mean(-2)
    print(xx.shape)
    xx = nn.Flatten()(xx)
    print(xx.shape)
    print('%%%')
    # print('\033[01;31m''After Unflatten''\033[0m')
    xx = nn.Unflatten(1, (64, 2))(xx)
    xx = xx.unsqueeze(-1)
    xx = nn.Linear(1, u)(xx)
    xx = xx.transpose(-1, -2)
    print(xx.shape)
    print('\033[01;31m''After UnMax''\033[0m')
    xx = nn.ConvTranspose2d(64, 64, (2, 2), stride=2, padding=(0, 0))(xx)
    print(xx.shape)
    xx = nn.ConvTranspose2d(64, 48, (2, 2), stride=2, padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)
    xx = nn.ConvTranspose2d(48, 48, (2, 2), stride=2, padding=(0, 0))(xx)
    print(xx.shape)
    xx = nn.ConvTranspose2d(48, 24, (2, 2), stride=2, padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)
    xx = nn.ConvTranspose2d(24, 24, (2, 2), stride=2, padding=(0, 0))(xx)
    print(xx.shape)
    xx = nn.ConvTranspose2d(24, 12, (2, 2), stride=2, padding=(0, 0))(xx)
    xx = nn.ReLU()(xx)
    print(xx.shape)
    # print('\033[01;31m''After ConvTranspose2d''\033[0m')
    # xx = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=(1, 1))(xx)
    # print(xx.shape)
    # xx = nn.ReLU()(xx)
    # print('\033[01;31m''After UnMax''\033[0m')
    # xx = nn.ConvTranspose2d(24, 24, (2, 2), stride=2, padding=(0, 0))(xx)
    # print(xx.shape)
    # print('\033[01;31m''After ConvTranspose2d''\033[0m')
    # xx = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=(1, 1))(xx)
    # print(xx.shape)
    # xx = nn.ReLU()(xx)
    # print('\033[01;31m''After UnMax''\033[0m')
    # xx = nn.ConvTranspose2d(12, 12, (2, 2), stride=2, padding=(0, 0))(xx)
    # print(xx.shape)
    # print('\033[01;31m''After ConvTranspose2d''\033[0m')
    # xx = nn.ConvTranspose2d(12, 1, 4, stride=2, padding=(1, 1))(xx)
    # print(xx.shape)
    # print('\033[01;31m''After Tanh''\033[0m')
    # xx = nn.Tanh()(xx)
    # print(xx.shape)