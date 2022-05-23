#!python
# region IMPORT --------------------------------------
if "preload": 
    ModelName = None; "ssh://lab531/storage/LabJob/Projects/RemakeAudioSegs/BornAgainAudioAE/prescripts.py"; BSZ = None; LR = None; WORKERS = None; TOTAL = None; EPOCHS = None; FeatBSZ = None; verbose = None; FILEROOT = __import__("os").path.dirname(__import__("os").path.realpath(__file__)); f = open(FILEROOT + '/prescripts.py'); exec(f.read()); f.close(); import os; os.environ['KALDI_ROOT'] = '/CHANGE_ME_IF_YOU_WANT'
from pathlib import Path; import pickle as pkl, glob, time
from tqdm import tqdm; import numpy as np; import pandas as pd
from torchaudio.datasets.librispeech import LIBRISPEECH
import torch, torch.nn as nn, torchaudio, s3prl.hub as hub
from src.datasets import MyDataset
from src.datasets import MyFeatDataset
from src.datasets import collate_fn_fast
from src.datasets import collate_fn_feat
from src.models import MyAutoencoder
from src.models import *
# endregion ------------------------------------------

def feature_extraction(libridata):
    FEATURE_NAME = "fbank"
    
    device = 'cuda' # or cpu
    # s3prl_root = Path('/storage/LabJob/LearnSplendid/s3prl')
    s3prl_root = Path(FILEROOT + '/''imppaths/s3prl')
    config_path = s3prl_root / f's3prl/upstream/baseline/{FEATURE_NAME}.yaml'
    extracter = getattr(hub, 
        'baseline_local')(model_config=config_path).to(device)
    bsz_dts = FeatBSZ
    
    FOLDER = "data/fbank"
    Path(FOLDER).mkdir(parents=True, exist_ok=True) 
    wavs = []; ids = []
    txts = {}
    
    for ni, thing in enumerate(tqdm(libridata)):
        it, sr, txt, numa, numb, numc = thing
        item_id = (
            # f"{FEATURE_NAME}_"
            f"{ni:06d}__"
            f"{numa:06d}_{numb:06d}_{numc:06d}")
        assert (sr == SAMPLE_RATE), "Sample rate different!"
        with open(FOLDER + '/'"text__" + item_id + '.txt', 'w') as f:
            print(txt, file=f)
        wavs.append(it.to(device))
        ids.append(FEATURE_NAME + '__' + item_id + '.tnsr')
        if ni % bsz_dts == bsz_dts - 1:
            with torch.no_grad():
                [feats] = extracter(wavs)["hidden_states"]
                for _id, _feat in zip(ids, feats):
                    torch.save(_feat.cpu(), FOLDER + '/' + _id)
            wavs = []; ids = []
    #################
    if len(wavs) > 0:
        with torch.no_grad():
            [feats] = extracter(wavs)["hidden_states"]
            self.feats.extend(feats)
        wavs = []; ids = []

if __name__ == "__main__":
    SAMPLE_RATE = 16_000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # libridata = LIBRISPEECH(
    #     root=FILEROOT + "/""imppaths/corpus", 
    #     url="train-clean-100")

    # feature_extraction(libridata)

    # dst = MyDataset(libridata, 
    #     FILEROOT + "/""imppaths/spectrograms")

    dst = MyFeatDataset(FILEROOT + "/""../data/fbank", workers=WORKERS)

    loader = torch.utils.data.DataLoader(
        dataset=dst,
        batch_size=BSZ,
        collate_fn=collate_fn_feat,
        num_workers=WORKERS,
    ); assert len(loader) > 0, 'Empty dataloader?'


    # for b in loader: break
    # print('from IPython import embed; embed()'); breakpoint()

    model = eval(ModelName).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

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
            x_resized = input_x[..., :recon_x.size(-1)].to(device)
            a_resized = attns_x[..., :recon_x.size(-1)].to(device)
            loss = criterion(
                recon_x * a_resized,
                x_resized * a_resized,
            )
            loss.backward()
            optimizer.step()
            total_loss += (batch_loss := loss.item()) * len(input_x)
            pbar.set_postfix({'batch_loss': "[%7.3lf]"[:7] % round(batch_loss, 3)})  # FIXME
        print(total_loss / (
            effective_total_size := min(
                tqdmTOTAL * BSZ + BSZ, 
                len(loader.dataset))))
