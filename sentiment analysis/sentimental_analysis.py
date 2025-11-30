# -----------------------------------------------------
# SENTIMENT ANALYSIS TOOL (Matches Question Requirements)
# -----------------------------------------------------

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------
# 1. Load labeled text dataset
# ---------------------------------
def load_data(path):
    df = pd.read_csv(path)      # CSV must have columns: text, label
    return df["text"], df["label"]

# ---------------------------------
# 2. Preprocess text (cleaning)
# ---------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\\s]', '', text)             # remove special chars
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# ---------------------------------
# Load + preprocess entire dataset
# ---------------------------------
def preprocess_texts(texts):
    return [clean_text(t) for t in texts]

# ---------------------------------
# 3. Convert text → numeric features (TF-IDF)
# 4. Train a classifier (Naive Bayes)
# ---------------------------------
def train_model(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # -------------------------------
    # 5. Evaluate the model
    # -------------------------------
    preds = model.predict(X_test)
    print("\nModel Evaluation")
    print("-------------------------")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds, average='weighted'))

    return model, vectorizer

# ---------------------------------
# 6. Simple CLI for predictions
# ---------------------------------
def cli(model, vectorizer):
    print("\nSentiment Analysis CLI")
    print("Type 'exit' to stop.\n")

    while True:
        text = input("Enter text: ")
        if text.lower() == "exit":
            break

        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]

        print("Predicted Sentiment:", pred, "\n")

# ---------------------------------
# MAIN PROGRAM
# ---------------------------------
if __name__ == "__main__":
    print("Loading data....")
    texts, labels = load_data("sentiment_data.csv")

    print("Preprocessing....")
    cleaned_texts = preprocess_texts(texts)

    print("Training model....")
    model, vectorizer = train_model(cleaned_texts, labels)

    cli(model, vectorizer)
