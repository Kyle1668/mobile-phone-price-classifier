"""Used to create the neurel network"""
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    training_set = PhonePricesDataSet("./data/train.csv")
    test_set = PhonePricesDataSet("./data/test.csv")

    training_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)

    EPOCHS = 5
    min_valid_loss = np.inf
    model = PhonePriceClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(EPOCHS):
        train_loss = 0.0
        for data, labels in tqdm(training_dataloader):
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(training_dataloader)}"
        )

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in test_dataloader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()

            target = model(data)
            loss = criterion(target, labels)
            valid_loss = loss.item() * data.size(0)

        print(
            f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(training_dataloader)} \t\t Validation Loss: {valid_loss / len(test_dataloader)}"
        )
        if min_valid_loss > valid_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), "saved_model.pth")
