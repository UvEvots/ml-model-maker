from pathlib import Path
import numpy as np
import pandas as pd

# load data
PROJECT_ROOT = Path(__file__).resolve().parents[3]
data_path = PROJECT_ROOT / "data" / "housing_prices.csv"

df = pd.read_csv(data_path)

#HYPERPARAMETERS
learningRate = 0.01
batchSize = 64
Epochs = 25

# PARAMETERS - STARTING POINT
weight = 0
bias = 0

# Normalizing features
feature = df["square_feet"].to_numpy()
normalized = (feature - feature.mean() / feature.std())

df["square_feet"] = normalized

for epoch in range(Epochs):
    batch = df.sample(batchSize)

    feature = batch["square_feet"].to_numpy()
    predictions = feature * weight + bias

    label = batch["price_thousands"].to_numpy()

    mse = np.mean((label - predictions) ** 2)
    print(mse)


