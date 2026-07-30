import pandas as pd
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords (first time only)
nltk.download("stopwords")

# Stopwords
stop_words = set(stopwords.words("english"))


# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = " ".join(text.split())

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------
# Load Dataset
# -----------------------------
fake_df = pd.read_csv("dataset/fake.csv")
true_df = pd.read_csv("dataset/true.csv")

# Add Labels
fake_df["label"] = 0
true_df["label"] = 1

# Merge Dataset
news_df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle Dataset
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean Text
news_df["text"] = news_df["text"].apply(clean_text)

# -----------------------------
# Information
# -----------------------------
print("=" * 50)
print("Dataset Shape :", news_df.shape)

print("\nLabel Count")
print(news_df["label"].value_counts())

print("\nSample Cleaned News\n")
print(news_df["text"].head())


# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(news_df["text"])

y = news_df["label"]

print("\nFeature Matrix Shape :", X.shape)


# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)


# -----------------------------
# Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000)

print("\nTraining Model...")
model.fit(X_train, y_train)

print("Model Training Completed Successfully!")


# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy : {accuracy * 100:.2f}%")


# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Location : model/model.pkl")
print("Location : model/vectorizer.pkl")