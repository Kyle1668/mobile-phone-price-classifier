"""Used to create the neurel network"""
from tqdm import tqdm
import torch
import pandas as pd
from torch import nn, optim
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
    """A feed-forward classification network for predicting phone price range"""

    def __init__(self, dropout_percent=0.1):
        super(PhonePriceClassifier, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(100, 4),
            # nn.Softmax(1)
        )

    def forward(self, input_vector):
        """Make an inference

        Args:
            x (Tensor): A 20 dimensional tensor

        Returns:
            _type_: _description_
        """
        logits = self.linear_relu_stack(input_vector)
        return logits


def train_model(model: PhonePriceClassifier, training_dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss):
    model.train()
    running_loss = 0.0

    for features, labels in tqdm(training_dataloader, "Training"):
        loss = train_batch(model, features, labels, optimizer, criterion)
        running_loss += loss * features.size(0)

    mean_loss = running_loss / len(training_dataloader)
    print(f"\nTrain Mean Loss = {mean_loss}\n")


def train_batch(model: PhonePriceClassifier, features: torch.Tensor, labels: torch.Tensor, optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss) -> float:
    # Set all the gradients to zero allowing the optimization to start with a blank slate.
    optimizer.zero_grad()

    # Perform inferences on the training batch
    logits = model(features)

    # Calculate the average loss across all the inferences
    loss = criterion(logits, labels)

    # Calculate all the necessary gradient changes using back progegation
    loss.backward()

    # Update the model parameters based off the gradient calculated during back propegation
    optimizer.step()

    return loss.item()


def evaluate_model(model: PhonePriceClassifier, test_dataloader: DataLoader, criterion: nn.CrossEntropyLoss):
    model.eval()
    running_loss = 0.0
    correct_inferences = 0

    for features, labels in tqdm(test_dataloader, "Test"):
        with torch.no_grad():
            logits = model(features)

        loss = criterion(logits, labels)
        running_loss += loss.item() * features.size(0)
        correct_inferences += (torch.argmax(logits, 1) == labels).int().sum().item()

    mean_loss = running_loss / len(test_dataloader.dataset)
    accuracy = 100 * (correct_inferences / len(test_dataloader.dataset))
    print(f"\nTest Mean Loss = {mean_loss} | Accuracy = {accuracy}%\n")


def main():
     # Set up data for feeding into our model during training and evaluation
    test_set = PhonePricesDataSet("./data/test.csv")
    training_set = PhonePricesDataSet("./data/train.csv")
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)
    training_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)

    # Init the training setup
    epochs = 100
    model = PhonePriceClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch_counter in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch_counter} ----\n")
        train_model(model, training_dataloader, optimizer, criterion)
        evaluate_model(model, test_dataloader, criterion)


if __name__ == "__main__":
    main()
