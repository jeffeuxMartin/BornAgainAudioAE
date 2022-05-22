BSZ = 320
LR = 2e-5

import sys
exec(sys.argv[1])
print(f"{BSZ = }"' 'f"{LR = }")

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import librosa

from multiprocessing import Pool
import time



dataset = (torchaudio
               .datasets
               .librispeech
               .LIBRISPEECH)(
    root="/diskfile/NewCorpus"
    "/LibriSpeech", 
    url="train-clean-100")

import torchaudio.transforms as T

n_fft = 2048 ###
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = lambda sample_rate: T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="constant",  ###
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    mel_scale="slaney",  #????
)

mel_torch = lambda y, sr: mel_spectrogram(sr)(y)

def bettermel(item):
    audio, sr, *other = item
    mel = mel_torch(audio, sr)
    return mel, other

class _MelDataset(Dataset):
    def __init__(self, libri):
        self.libri = libri
    def getspec(self, idx_nme):
        return torch.load(idx_nme)
    def __getitem__(self, index):
        out = self.libri[index]
        wave, sr, txt, numa, numb, numc = out
        mel = self.getspec(f'Upstea/'f'{index}__{numa}_{numb}_{numc}.tnsr')
        return mel
    def __len__(self):
        return len(self.libri)

class MelDataset(Dataset):
    def __init__(self, libri):
        self.libri = libri
        # self.mels = for i in range(len(self.libri))
    def getspec(self, idx_nme):
        return torch.load(idx_nme)
    def __getitem__(self, index):
        out = self.libri[index]
        wave, sr, txt, numa, numb, numc = out
        mel = self.getspec(f'Upstea/'f'{index}__{numa}_{numb}_{numc}.tnsr')
        return mel
    def __len__(self):
        return len(self.libri)

def collate_fn(batch):
    melspecs = pad_sequence(
        ([i.squeeze().T
        # ([i.transpose(0, -1).squeeze(-1) 
        #    .transpose(-1, -2)
          for i in batch]), batch_first=True)
    # melspecs = melspecs.unsqueeze(1)
    melspecs = melspecs.transpose(1, 2)
    return melspecs

loader = torch.utils.data.DataLoader(
    dataset=MelDataset(dataset),
    batch_size=BSZ,
    collate_fn=collate_fn,
    num_workers=4,
)

        
class __MelDataset(Dataset):
    def audio_to_mel(self, inputs):
        audio, sr = inputs
        mel = librosa.feature.melspectrogram(
            y=audio.numpy(), sr=sr)
        return mel

    def __init__(self, libridataset):
        self.audio_srs = []
        self.mels = []
        self.others = []
        for item in tqdm(libridataset):
            audio, sr, *other = item
            # self.mels.append(
            #     librosa.feature.melspectrogram(
            #         y=audio.numpy(), sr=sr))
            self.audio_srs.append((audio, sr))
            self.others.append(other)
        with Pool(4) as p:
            self.mels = list(tqdm(p.imap(
                mel_torch, self.audio_srs), 
                    total=len(self.audio_srs)))

    def __len__(self):
        return len(self.audio_srs)
        
    def __getitem__(self, idx):
        return (self.mels[idx])
            
def __collate_fn(batch):
    melspecs = pad_sequence(
        ([torch.tensor(i).transpose(0, -1).squeeze(-1) 
          for i in batch]), batch_first=True)
    melspecs = melspecs.unsqueeze(1)
    return melspecs

def __old_collate_fn(batch):
    audios, srs, *others = list(zip(*batch))
    # audios = [torchaudio.compliance.kaldi.fbank(a)
    #     for a in audios]
    melspecs = [librosa.feature.melspectrogram(
        y=audio.numpy(), sr=sr)
        for audio, sr in zip(audios, srs)]
    melspecs = pad_sequence(
        ([torch.tensor(i).transpose(0, -1).squeeze(-1) 
          for i in melspecs]), batch_first=True)
    melspecs = melspecs.unsqueeze(1)
    
    return melspecs, others

class Conv_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool2d = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.conv1 = nn.Conv1d( 128,  512, 32, stride=8, padding=1)
        self.conv2 = nn.Conv1d( 512, 1024, 16, stride=4, padding=1)
        self.conv3 = nn.Conv1d(1024, 2048,  8, stride=4, padding=1)
        
        self.conv_v3 = nn.ConvTranspose1d(2048, 1024,  8, stride=4, padding=1)
        self.conv_v2 = nn.ConvTranspose1d(1024,  512, 16, stride=4, padding=1)
        self.conv_v1 = nn.ConvTranspose1d( 512,  128, 32, stride=8, padding=1)

    def forward(self, x):
        # print('\033[01;31m'f"{x.shape[-1] * x.shape[-2] = }"'\033[0m')
        x = self.relu(self.conv1(x))
        # print('\033[01;31m'f"{x.shape[-1] * x.shape[-2] = }"'\033[0m')
        x = self.relu(self.conv2(x))
        # print('\033[01;31m'f"{x.shape[-1] * x.shape[-2] = }"'\033[0m')
        x = self.relu(self.conv3(x))
        z = x
        # print('\033[01;31m'f"{z.shape[-1] * z.shape[-2] = }"'\033[0m')
        y = z
        y = self.relu(self.conv_v3(y))
        # print('\033[01;31m'f"{y.shape[-1] * y.shape[-2] = }"'\033[0m')
        y = self.relu(self.conv_v2(y))
        # print('\033[01;31m'f"{y.shape[-1] * y.shape[-2] = }"'\033[0m')
        y = self.tanh(self.conv_v1(y))
        # print('\033[01;31m'f"{y.shape[-1] * y.shape[-2] = }"'\033[0m')
        
        return z, y

class MyLstm(nn.Module):
    def __init__(self, input_size=128, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            batch_first=True
        )
    
    def forward(self, x, hiddens=None):
        if hiddens is None:
            output, (h_state, c_state) = self.lstm(x)
        else:
            output, (h_state, c_state) = self.lstm(x, hiddens)
        return output, (h_state, c_state)

class Lstm_autoencoder(nn.Module):
    def __init__(self):
        super(Lstm_autoencoder, self).__init__()
        self.encoder = MyLstm()
        self.decoder = MyLstm()

    def forward(self, x):
        seqlen = x.size(1)
        _, (h, c) = self.encoder(x)
        recon_x = self.decoder()
        z = None  # FIXME
        return z, recon_x

if 0:
    loader = torch.utils.data.DataLoader(
        dataset=MelDataset(dataset),
        batch_size=BSZ,
        collate_fn=collate_fn,
        num_workers=8,
    )

model = Conv_autoencoder()
criterion = nn.MSELoss()
opt = torch.optim.Adam(
    params=model.parameters(),
    lr=LR,
)
device = "cuda"

model = model.to(device)

# clock = 0
for epoch in range(10):
    model.train()
    total_loss = 0.
    pbar = tqdm(loader)
    for batch in pbar:
        # if clock > 3: break
        opt.zero_grad()
        # x, *_ = batch
        x = batch
        x = x.to(device)
        latent, recon_x = model(x)
        loss = criterion(
            recon_x,
            x[..., :recon_x.size(-1)]
        )
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(x)
        pbar.set_postfix({
            'loss': "[%9.3lf]" % round(loss.item(), 3)})  # FIXME
        # clock += 1
    print(total_loss / len(loader.dataset))


print(f"{latent.shape = }")
print(f"{recon_x.shape = }")

exit()

raise AssertionError
librosa.feature.melspectrogram(
    
)
# region #############@@@@@@@@@@@@2

LLL = 100

n_fft = 2048 ###
win_length = None
hop_length = 512
n_mels = 128

import torchaudio.transforms as T
mel_spectrogram = lambda sample_rate: T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="constant",  ###
    power=2.0,
    norm='slaney',
    onesided=True,
    n_mels=n_mels,
    mel_scale="slaney",  #????
)

mel_torch = lambda y, sr: mel_spectrogram(sr)(y)
if 1:
    def get_speech_sample(*, resample=None):
      return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)
    # !wget https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav
    def _get_sample(path, resample=None):
      effects = [
        ["remix", "1"]
      ]
      if resample:
        effects.extend([
          ["lowpass", f"{resample // 2}"],
          ["rate", f'{resample}'],
        ])
      return torchaudio.sox_effects.apply_effects_file(path, effects=effects)
    SAMPLE_WAV_SPEECH_PATH = './Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
y, sr = get_speech_sample()
Sam = dict(y=y, sr=sr)
Samp = dict(y=y.numpy(), sr=sr)

librosa.feature.melspectrogram(
    **Samp
)

mel_lib = librosa.feature.melspectrogram

mel_torch(**Sam).numpy() - mel_lib(**Samp)

i = 0
uU, iV = 0., 0.
for datasetdataset in dataset:
    if i > LLL: break
    y, sr, *o = datasetdataset
    Sam = dict(y=y, sr=sr)
    Samp = dict(y=y.numpy(), sr=sr)
    D = mel_torch(**Sam).numpy() - mel_lib(**Samp)
    U, V = ((D.min(), D.max()))
    uU += U
    iV += V
    print((U, V))
    i += 1
print()
print(uU / LLL, iV / LLL)
# endregion #############@@@@@@@@@@@@2
