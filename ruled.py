#!python

# region
BSZ = 32
LR = 2e-5
WORKERS = 8
FeatBSZ = 2048

import sys
exec(sys.argv[1] if len(sys.argv) > 1 else '')
print(f"{BSZ = }"
' 'f"{LR = }"
' 'f"{WORKERS = }"
' 'f"{FeatBSZ = }"
)
print()

from pathlib import Path
import torch
import s3prl.hub as hub
import pickle as pkl

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


device = 'cuda' # or cpu
# s3prl_root = Path('/storage/LabJob/LearnSplendid/s3prl')
s3prl_root = Path('imppaths/s3prl')
config_path = s3prl_root / 's3prl/upstream/baseline/mfcc.yaml'
extracter = getattr(hub, 
    'baseline_local')(model_config=config_path).to(device)

# wavs = [torch.zeros(160000, dtype=torch.float).to(device) for _ in range(16)]
# with torch.no_grad():
#     mfcc = extracter(wavs)["hidden_states"]

dataset = (torchaudio.datasets.librispeech.LIBRISPEECH)(
    # root="/diskfile/NewCorpus",
    root="imppaths/corpus",
    url="train-clean-100")

SAMPLE_RATE = 16_000

class FeatDataset(Dataset):
    def __init__(self, dts) -> None:
        self.dts = dts
        bsz_dts = FeatBSZ
        self.feats = []
        self.txts = []
        wavs = []
        for ni, thing in enumerate(tqdm(self.dts)):
            it, sr, txt, *_ = thing
            self.txts.append(txt)
            assert (sr == SAMPLE_RATE), "Sample rate different!"
            wavs.append(it.to(device))
            if ni % bsz_dts == bsz_dts - 1:
                with torch.no_grad():
                    [mfcc] = extracter(wavs)["hidden_states"]
                    self.feats.extend(mfcc)
                wavs = []
        #################
        if len(wavs) > 0:
            with torch.no_grad():
                [mfcc] = extracter(wavs)["hidden_states"]
                self.feats.extend(mfcc)
            wavs = []
        # ~~~~~~~~~~~~~~

    def __len__(self):
        return len(self.dts)

    def __getitem__(self, idx):
        return self.feats[idx], self.txts[idx]
        
def collate_fn(batch):
    feats, txts = list(zip(*batch))
    return pad_sequence(feats, batch_first=True), txts

def collate_fn_slow(batch):
    wavs, sr, txt, numa, numb, numc = list(zip(*batch))
    assert all(_sr == SAMPLE_RATE for _sr in sr), "Sample rate different!"
    with torch.no_grad():
        [mfcc] = extracter(wavs)["hidden_states"]
    return mfcc, txt
    
    
# endregion
CACHE_NAME = 'featdtst.pkl'
if Path(CACHE_NAME).is_file():
    with open(CACHE_NAME, 'rb') as f:
        featdtst = pkl.load(f)
else:
    featdtst = FeatDataset(dataset)
    with open(CACHE_NAME, 'wb') as f:
        pkl.dump(featdtst, f)

loader = torch.utils.data.DataLoader(
    dataset=featdtst,
    batch_size=BSZ,
    collate_fn=collate_fn,
    num_workers=WORKERS,
)

class Conv_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool2d = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.conv1 = nn.Conv1d( 39, 128, 32, stride=8, padding=1)
        self.conv2 = nn.Conv1d(128, 256, 16, stride=8, padding=1)
        self.conv3 = nn.Conv1d(256, 512,  8, stride=4, padding=1)
        
        self.conv_v3 = nn.ConvTranspose1d(512, 256,  8, stride=4, padding=1)
        self.conv_v2 = nn.ConvTranspose1d(256, 128, 16, stride=8, padding=1)
        self.conv_v1 = nn.ConvTranspose1d(128,  39, 32, stride=8, padding=1)

    def forward(self, x):
        x = x.transpose(-1, -2)
        # print('\n''\033[01;33m'f"{x.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{x.shape[-1] * x.shape[-2] = }"'\033[0m')
        x = self.relu(self.conv1(x))
        # print('\033[01;33m'f"{x.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{x.shape[-1] * x.shape[-2] = }"'\033[0m')
        x = self.relu(self.conv2(x))
        # print('\033[01;33m'f"{x.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{x.shape[-1] * x.shape[-2] = }"'\033[0m')
        x = self.relu(self.conv3(x))
        z = x
        # print('\033[01;33m'f"{z.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{z.shape[-1] * z.shape[-2] = }"'\033[0m')
        y = z
        y = self.relu(self.conv_v3(y))
        # print('\033[01;33m'f"{y.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{y.shape[-1] * y.shape[-2] = }"'\033[0m')
        y = self.relu(self.conv_v2(y))
        # print('\033[01;33m'f"{y.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{y.shape[-1] * y.shape[-2] = }"'\033[0m')
        y = self.tanh(self.conv_v1(y))
        # print('\033[01;33m'f"{y.shape = }"'\033[0m', end='\n\t')
        # print('\033[01;32m'f"{y.shape[-1] * y.shape[-2] = }"'\033[0m')
        y = y.transpose(-1, -2)
        
        return z, y

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
        x, t = batch
        x = x.to(device)
        latent, recon_x = model(x)
        # print('\033[0;35m'f"{x.shape = }"'\033[0m')
        # print('\033[0;35m'f"{recon_x.shape = }"'\033[0m')
        loss = criterion(
            recon_x,
            x[..., :recon_x.size(-2), :]
        )
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(x)
        pbar.set_postfix({
            'loss': "[%9.3lf]" % round(loss.item(), 3)})  # FIXME
        # clock += 1
    print(total_loss / len(loader.dataset))


# print(f"{latent.shape = }")
# print(f"{recon_x.shape = }")
