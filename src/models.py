from typing import List
import numpy as np

import torch
import torch.nn as nn

class WeirdAutoencoder(nn.Module):  # Conv_autoencoder
    def __init__(self, feattype='fbank'):
        filts, kers, strs, pads = [], [], [], []
        super().__init__()
        if   feattype == 'spec':
            filts, kers, strs, pads = [128, 512, 1024, 2048], [8, 4, 2], [2, 2, 2], [1, 1, 1]
        elif feattype == 'mfcc':
            filts, kers, strs, pads = [39, 128, 256, 512], [32, 16, 8], [8, 8, 4], [1, 1, 1]
        elif feattype == 'fbank':
            filts, kers, strs, pads = [80 * 3, 512, 1024, 2048], [8, 4, 2], [2, 2, 2], [1, 1, 1]
        self.maxpool2d = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.conv1 = nn.Conv1d(filts[0], filts[0 + 1], kers[0], stride=strs[0], padding=pads[0])
        self.conv2 = nn.Conv1d(filts[1], filts[1 + 1], kers[1], stride=strs[1], padding=pads[1])
        self.conv3 = nn.Conv1d(filts[2], filts[2 + 1], kers[2], stride=strs[2], padding=pads[2])
        
        self.conv_v3 = nn.ConvTranspose1d(filts[2 + 1], filts[2], kers[2], stride=strs[2], padding=pads[2])
        self.conv_v2 = nn.ConvTranspose1d(filts[1 + 1], filts[1], kers[1], stride=strs[1], padding=pads[1])
        self.conv_v1 = nn.ConvTranspose1d(filts[0 + 1], filts[0], kers[0], stride=strs[0], padding=pads[0])
        
        in_channels   = 128 if feattype == 'spec' else (39 if feattype == 'mfcc' else 80 * 3)
        channels      = 768 if feattype == 'spec' else (39 if feattype == 'mfcc' else 360)
        embedding_dim =  64 if feattype == 'spec' else (39 if feattype == 'mfcc' else 128)
        
        self.encoder = nn.Sequential(
            self.conv1, self.maxpool2d, self.relu,
            self.conv2, self.maxpool2d, self.relu,
            self.conv3, self.maxpool2d, self.relu,
        )
        self.decoder = nn.Sequential(
            self.conv_v3, self.relu,
            self.conv_v2, self.relu,
            self.conv_v1, self.tanh,
        )

    def forward(self, x):
        # print(x.shape)
        z = self.encoder(x)
        # print(x.shape)
        x = self.decoder(z)
        # print(x.shape)
        return z, x

class MyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(80 * 3,  360, 12, 6, 0, bias=False), nn.BatchNorm1d( 360), nn.ReLU(True),
            nn.Conv1d(   360,  480,  6, 4, 0, bias=False), nn.BatchNorm1d( 480), nn.ReLU(True),
            nn.Conv1d(   480,  720,  6, 4, 0, bias=False), nn.BatchNorm1d( 720), nn.ReLU(True),
            nn.Conv1d(   720, 1080,  3, 3, 0, bias=False), nn.BatchNorm1d(1080), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1080,    720,  3, 3, 0, bias=False), nn.BatchNorm1d(   720), nn.ReLU(True),
            nn.ConvTranspose1d( 720,    480,  6, 4, 0, bias=False), nn.BatchNorm1d(   480), nn.ReLU(True),
            nn.ConvTranspose1d( 480,    360,  6, 4, 0, bias=False), nn.BatchNorm1d(   360), nn.ReLU(True),
            nn.ConvTranspose1d( 360, 80 * 3, 12, 6, 0, bias=False), nn.BatchNorm1d(80 * 3), nn.Tanh(    ),
        )
        

    def forward(self, x):
        # print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        x = self.decoder(z)
        return z, x

class BogiAutoencoder(nn.Module):
    def __init__(self, in_channels, channels, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels,      channels, 3, 1, 0, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.Conv1d(   channels,      channels, 3, 1, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.Conv1d(   channels,      channels, 4, 2, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.Conv1d(   channels,      channels, 3, 1, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.Conv1d(   channels,      channels, 3, 1, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True),
            nn.Conv1d(   channels, embedding_dim, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim,    channels, 1),                   nn.BatchNorm1d(   channels), nn.ReLU(True),
            nn.ConvTranspose1d(     channels,    channels, 3, 1, 1, bias=False), nn.BatchNorm1d(   channels), nn.ReLU(True),
            nn.ConvTranspose1d(     channels,    channels, 3, 1, 1, bias=False), nn.BatchNorm1d(   channels), nn.ReLU(True),
            nn.ConvTranspose1d(     channels,    channels, 4, 2, 1, bias=False), nn.BatchNorm1d(   channels), nn.ReLU(True),
            nn.ConvTranspose1d(     channels,    channels, 3, 1, 1, bias=False), nn.BatchNorm1d(   channels), nn.ReLU(True),
            nn.ConvTranspose1d(     channels, in_channels, 3, 1, 0, bias=False), nn.BatchNorm1d(in_channels), nn.Tanh(    ),
        )
        

    def forward(self, x):
        # print(x.shape)
        z = self.encoder(x)
        # print(z.shape)
        # assert np.prod(z.shape[1:]) / np.prod(x.shape[1:]) < 1, f"No compression! {x.shape} --> {z.shape}"
        x = self.decoder(z)
        return z, x

