#!python

# region
BSZ = 32
LR = 2e-4
WORKERS = 8
TOTAL = -1
EPOCHS = 10
FeatBSZ = 2048
verbose = False

import sys
exec(sys.argv[1] if len(sys.argv) > 1 else '')
print(f"{BSZ = }"
' 'f"{LR = }"
' 'f"{WORKERS = }"
# ' 'f"{FeatBSZ = }"
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
# endregion

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
    
import glob
class MyDataset(Dataset):
    def __init__(self, dst, paths):
        self.dst = dst
        self.paths = glob.glob(f"{paths}/*.tnsr")
        self.paths = [
            (i.split('/')[-1], i) for i in self.paths]
        self.paths = [(int(i.split('__')[0]), j) 
            for i, j in self.paths]
        self.paths.sort()
        
        assert all([
            idx == num for idx, (num, item) in 
                enumerate(self.paths)])
    def __len__(self): return len(self.dst)
    def __getitem__(self, idx): 
        return (
            self.paths[idx][-1],
            self.dst[idx][2], 
        )
    
dst = MyDataset(
    dataset,
    # "/storage/LabJob/Projects/RemakeAudioSegs/Upstea",
    "imppaths/spectrograms",
)

def collate_fn_fast(batch):
    feats, txts = list(zip(*batch))
    feats = [torch.load(i).squeeze(0).T for i in feats]
    ones = [torch.ones_like(i) for i in feats]
    # return feats, txts
    return (
        pad_sequence(feats, batch_first=True), 
        pad_sequence(ones, batch_first=True), 
        txts)

loader = torch.utils.data.DataLoader(
    dataset=dst,
    batch_size=BSZ,
    collate_fn=collate_fn_fast,
    num_workers=WORKERS,
)

# endregion
if 0:
    CACHE_NAME = 'featdtst.pkl'
    if Path(CACHE_NAME).is_file():
        with open(CACHE_NAME, 'rb') as f:
            featdtst = pkl.load(f)
    else:
        featdtst = FeatDataset(dataset)
        # with open(CACHE_NAME, 'wb') as f:
        #     pkl.dump(featdtst, f)

    loader = torch.utils.data.DataLoader(
        dataset=featdtst,
        batch_size=BSZ,
        collate_fn=collate_fn,
        num_workers=WORKERS,
    )
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
        
    
    
model = Conv_autoencoder('spec')
criterion = nn.MSELoss()
opt = torch.optim.Adam(
    params=model.parameters(),
    lr=LR,
)
device = "cuda"

model = model.to(device)

# clock = 0
tqdmTOTAL = len(loader) if TOTAL == -1 else TOTAL
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.
    pbar = tqdm(loader, total=tqdmTOTAL)
    for bidx, batch in enumerate(pbar):
        if bidx >= tqdmTOTAL: break
        # if clock > 3: break
        opt.zero_grad()
        # x, *_ = batch
        x, a, t = batch
        x = x.to(device)
        a = a.to(device)
        latent, recon_x = model(x)
        # print('\033[0;35m'f"{x.shape = }"'\033[0m')
        # print('\033[0;35m'f"{recon_x.shape = }"'\033[0m')
        xr = x[..., :recon_x.size(-2), :]
        ar = a[..., :recon_x.size(-2), :]
        loss = criterion(
            recon_x * ar,
            xr * ar,
        )
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(x)
        pbar.set_postfix({
            'loss': "[%9.3lf]" % round(loss.item(), 3)})  # FIXME
        # clock += 1
        break
    print(total_loss / len(loader.dataset))


# print(f"{latent.shape = }")
# print(f"{recon_x.shape = }")
