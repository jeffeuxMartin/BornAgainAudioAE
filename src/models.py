import torch
import torch.nn as nn

class Conv_autoencoder(nn.Module):
    def __init__(self, feattype='spec'):
        super().__init__()
        if feattype == 'spec':
            filts, kers, strs, pads = [128, 512, 1024, 2048], [8, 4, 2], [2, 2, 2], [1, 1, 1]
        elif feattype == 'mfcc':
            filts, kers, strs, pads = [39, 128, 256, 512], [32, 16, 8], [8, 8, 4], [1, 1, 1]
        self.maxpool2d = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.conv1 = nn.Conv1d(filts[0], filts[0 + 1], kers[0], stride=strs[0], padding=pads[0])
        self.conv2 = nn.Conv1d(filts[1], filts[1 + 1], kers[1], stride=strs[1], padding=pads[1])
        self.conv3 = nn.Conv1d(filts[2], filts[2 + 1], kers[2], stride=strs[2], padding=pads[2])
        
        self.conv_v3 = nn.ConvTranspose1d(filts[2 + 1], filts[2], kers[2], stride=strs[2], padding=pads[2])
        self.conv_v2 = nn.ConvTranspose1d(filts[1 + 1], filts[1], kers[1], stride=strs[1], padding=pads[1])
        self.conv_v1 = nn.ConvTranspose1d(filts[0 + 1], filts[0], kers[0], stride=strs[0], padding=pads[0])
        
        in_channels = 128 if feattype == 'spec' else 39
        channels = 768 if feattype == 'spec' else 39
        embedding_dim = 64 if feattype == 'spec' else 39
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, embedding_dim, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(channels, in_channels, 3, 1, 0, bias=False),
            # nn.BatchNorm1d(in_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        # print(x.shape)
        x = x.transpose(-1, -2)
        z = self.encoder(x)
        # print(x.shape)
        x = self.decoder(z)
        # print(x.shape)
        x = x.transpose(-1, -2)
        return z, x
