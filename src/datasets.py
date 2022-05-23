from pathlib import Path
import glob
from multiprocessing import Pool

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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
    
def collate_fn_fast(batch):
    feats, txts = list(zip(*batch))
    feats = [torch.load(i).squeeze(0).T for i in feats]
    ones = [torch.ones_like(i) for i in feats]
    # return feats, txts
    return (
        pad_sequence(feats, batch_first=True).transpose(-1, -2), 
        pad_sequence(ones, batch_first=True).transpose(-1, -2), 
        txts)

class MyFeatDataset(Dataset):
    def readtext(self, txtpath):
        with open(txtpath) as f:
            return (f.read().strip().lower())  # XXX: 不要寫死 lower
            
    
    def __init__(self, datapath, workers=0):
        self.workers = workers
        
        self.feats = sorted((Path(datapath)).glob("*.tnsr"))
        txtpaths = sorted((Path(datapath)).glob("*.txt"))
        if self.workers > 0:
            with Pool(self.workers) as p:
                self.txts = list(tqdm(p.imap(
                    self.readtext,
                    txtpaths, 
                    chunksize=1000),
                    total=len(txtpaths),
                    desc='Read texts...',
                ))
        else:    
            self.txts = [self.readtext(txtpath) 
                for txtpath in tqdm(txtpaths, desc='Read texts...')]

    def __len__(self):
        return len(self.feats)
        
    def __getitem__(self, idx):
        return (
            feat := torch.load(self.feats[idx]),
            torch.ones_like(feat),
            self.txts[idx],
        )

def collate_fn_feat(batch):
    # workers = 8
    feats, ones, txts = list(zip(*batch))
    # feats = [torch.load(i) for i in feats]  # workerfy! TODO: !!!
    # if workers > 0:
    #     with Pool(workers) as p:
    #         feats = list(tqdm(p.imap(
    #             torch.load,
    #             feats, 
    #             chunksize=workers), 
    #             total=len(feats),
    #             desc='Read batch...',
    #         ))
    # else:    
    #     feats = [torch.load(i) 
    #         for i in tqdm(feats, desc='Read batch...')]

    # ones = [torch.ones_like(i) for i in feats]
    return (
        # pad_sequence(feats, batch_first=True),
        pad_sequence(feats, batch_first=True).transpose(1, 2),
        # pad_sequence(ones, batch_first=True),
        pad_sequence(ones, batch_first=True).transpose(1, 2),
        txts)


# from multiprocessing import Pool
# workers = 8
# with Pool(workers) as p:
#     self.mels = list(tqdm(p.imap(
#         mel_torch, self.audio_srs,
#         chunksize=4), 
#             total=len(self.audio_srs)))
