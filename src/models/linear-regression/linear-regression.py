from pathlib import Path
import numpy as np
import pandas as pd

# load data
PROJECT_ROOT = Path(__file__).resolve().parents[3]
data_path = PROJECT_ROOT / "data" / "housing_prices.csv"

df = pd.read_csv(data_path)

print(df.head())

#HYPERPARAMETERS
learningRate = 0.01
batchSize = 64
Epochs = 25

