import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/train.csv")
data.head()


label_counts = {}

for label in data["price_range"].unique():
    label_counts[label] = (data["price_range"] == label).count()

plt.xlabel("Price Range")
plt.ylabel("Count Occurrences")
plt.bar(label_counts.keys(), label_counts.values())
plt.ion()