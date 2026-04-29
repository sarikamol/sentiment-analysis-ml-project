import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# keep negation words
stop_words.remove("not")
stop_words.remove("no")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

print("Dataset loaded")

df["review"] = df["review"].apply(preprocess)

# Convert text → numbers
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")

# Train model
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Test custom input
sample = ["This movie was amazing and fantastic"]
print("Prediction:", model.predict(vectorizer.transform(sample)))

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully")