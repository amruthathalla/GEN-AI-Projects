import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(url, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

# Split into training data (80%) and test data (20%), BEFORE doing anything else.
# This matters: the model must never see the test data until final evaluation,
# or our evaluation will be misleadingly optimistic.
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print(f"Training messages: {len(X_train)}")
print(f"Test messages: {len(X_test)}")

# Turn text into TF-IDF numerical features.
# max_features caps the vocabulary size to the 3000 most informative words,
# which keeps things fast and avoids overfitting to extremely rare words.
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)  # learn vocabulary AND transform training data
X_test_vec = vectorizer.transform(X_test)          # ONLY transform test data using that same vocabulary

print(f"\nFeature matrix shape (train): {X_train_vec.shape}")
print(f"Feature matrix shape (test): {X_test_vec.shape}")