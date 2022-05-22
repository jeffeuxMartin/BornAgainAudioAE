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
# endregion
# region
from pathlib import Path
import torch
# import s3prl.hub as hub
import pickle as pkl
import glob

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets.librispeech import LIBRISPEECH
import time

from src.datasets import MyDataset
from src.datasets import collate_fn_fast
from src.models import Conv_autoencoder
# endregion

SAMPLE_RATE = 16_000
device = 'cuda' # or cpu

dst = MyDataset(LIBRISPEECH(
    root="imppaths/corpus", url="train-clean-100"), 
    "imppaths/spectrograms")

loader = torch.utils.data.DataLoader(
    dataset=dst,
    batch_size=BSZ,
    collate_fn=collate_fn_fast,
    num_workers=WORKERS,
)

model = Conv_autoencoder('spec').to(device)
criterion = nn.MSELoss()
opt = torch.optim.Adam(params=model.parameters(), lr=LR)

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
        # break
    print(total_loss / len(loader.dataset))
