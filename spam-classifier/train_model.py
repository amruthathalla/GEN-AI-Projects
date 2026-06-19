import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
df = pd.read_csv(url, encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Model trained.")

# Quick sanity check: try it on a couple of made-up messages
sample_messages = [
    "Congratulations! You've won a free prize, click here to claim now!!!",
    "Hey, are we still meeting for lunch tomorrow?",
]
sample_vec = vectorizer.transform(sample_messages)
predictions = model.predict(sample_vec)

for msg, pred in zip(sample_messages, predictions):
    print(f"\nMessage: {msg}\nPredicted: {pred}")

# Save the trained model and vectorizer so we can reuse them later (e.g. in a small demo)
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModel and vectorizer saved.")