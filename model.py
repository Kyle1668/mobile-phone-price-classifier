import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class PhonePricesDataSet(Dataset):
    """Load the mobile click data set into a PyTorch representation"""

    def __init__(self, path: str):
        super(PhonePricesDataSet, self).__init__()
        frame = pd.read_csv(path)
        frame = frame.drop_duplicates().dropna()
        self.data = frame

    def __getitem__(self, index):
        phone_example: pd.Series = self.data.iloc[index]
        features_series = phone_example.drop("price_range")
        features_series = features_series.apply(lambda val: float(val))
        features_tensor = torch.Tensor(list(features_series.values))
        labels = int(phone_example["price_range"])
        return (features_tensor, labels)

    def __len__(self):
        return len(self.data)


class PhonePriceClassifier(nn.Module):
    """Some Information about PhonePriceClassifier"""

    def __init__(self):
        super(PhonePriceClassifier, self).__init__()

    def forward(self, x):

        return x


if __name__ == "__main__":
    training_set = PhonePricesDataSet("./data/train.csv")
    test_set = PhonePricesDataSet("./data/test.csv")

    training_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)
