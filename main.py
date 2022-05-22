#!python
# region
"ssh://lab531/storage/LabJob/Projects/RemakeAudioSegs/BornAgainAudioAE/prescripts.py"; BSZ=None;LR=None;WORKERS=None;TOTAL=None;EPOCHS=None;FeatBSZ=None;verbose=None;FILEROOT=__import__("os").path.dirname(__import__("os").path.realpath(__file__)); f = open(FILEROOT + '/prescripts.py'); exec(f.read()); f.close()
from pathlib import Path
import pickle as pkl
import glob
import time

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch, torch.nn as nn
import torchaudio
from torchaudio.datasets.librispeech import LIBRISPEECH
import s3prl.hub as hub

from src.datasets import MyDataset
from src.datasets import collate_fn_fast
from src.models import Conv_autoencoder
# endregion

SAMPLE_RATE = 16_000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dst = MyDataset(LIBRISPEECH(
    root=FILEROOT + "/""imppaths/corpus", url="train-clean-100"), 
    FILEROOT + "/""imppaths/spectrograms")

loader = torch.utils.data.DataLoader(
    dataset=dst,
    batch_size=BSZ,
    collate_fn=collate_fn_fast,
    num_workers=WORKERS,
); assert len(loader) > 0, 'Empty dataloader?'

model = Conv_autoencoder('spec').to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.
    for batch_idx, batch in enumerate(
        pbar := tqdm(loader, total=(
            tqdmTOTAL := (len(loader) 
                          if TOTAL == -1 else TOTAL)))
      ):
        if batch_idx >= tqdmTOTAL: break
        optimizer.zero_grad()
        input_x, attns_x, texts = batch
        latent, recon_x = model(input_x.to(device))
        if verbose: print('\033[0;35m'f"{input_x.shape = }"'\033[0m')
        if verbose: print('\033[0;35m'f"{recon_x.shape = }"'\033[0m')
        x_resized = input_x[..., :recon_x.size(-2), :].to(device)
        a_resized = attns_x[..., :recon_x.size(-2), :].to(device)
        loss = criterion(
            recon_x * a_resized,
            x_resized * a_resized,
        )
        loss.backward()
        optimizer.step()
        total_loss += (batch_loss := loss.item()) * len(input_x)
        pbar.set_postfix({'batch_loss': "%9.3lf" % round(batch_loss, 3)})  # FIXME
    print(total_loss / (
        effective_total_size := min(
            tqdmTOTAL * BSZ + BSZ, 
            len(loader.dataset))))
