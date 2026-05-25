import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print("\nTrain info:")
print(train.info())

print("\nMissing values in train:")
print(train.isnull().sum())

print("\nMissing values in test:")
print(test.isnull().sum())

print("\nTarget distribution:")
print(train['Drafted'].value_counts(normalize=True))
