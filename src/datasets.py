import glob

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
