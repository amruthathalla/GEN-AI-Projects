import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin-1")
print(df.shape)
print(df.columns.tolist())
print(df.head())