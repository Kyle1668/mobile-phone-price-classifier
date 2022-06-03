import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class PhonePricesDataSet(torch.utils.data.Dataset):
    """Some Information about PhonePricesDataSet"""
    def __init__(self):
        super(PhonePricesDataSet, self).__init__()

    def __getitem__(self, index):
        return 

    def __len__(self):
        return 