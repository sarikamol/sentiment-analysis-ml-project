import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Simple dataset (we will improve later)
data = {
    "text": [
        "I love this movie",
        "This is amazing",
        "I hate this",
        "Very bad experience",
        "So good and nice",
        "Worst ever"
    ],
    "sentiment": [
        "positive",
        "positive",
        "negative",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test prediction
sample = ["I really love this product"]
prediction = model.predict(vectorizer.transform(sample))
print("Prediction:", prediction)