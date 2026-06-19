import pandas as pd

url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(url, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

print(df.head())
print(f"\nTotal messages: {len(df)}")
print(df["label"].value_counts())