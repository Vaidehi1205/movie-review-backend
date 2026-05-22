import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("IMDB Dataset.csv")

# Convert sentiment to numeric
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['review'] = df['review'].apply(clean_text)

# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# ==============================
# TF-IDF VECTORIZER
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# ==============================
# TRAIN MODELS
# ==============================
print("Training models...")

logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


# ==============================
# CREATE MODELS FOLDER
# ==============================
os.makedirs("models", exist_ok=True)

# ==============================
# SAVE MODELS
# ==============================
joblib.dump(logistic_model, "models/logistic.pkl")
joblib.dump(svm_model, "models/svm.pkl")
joblib.dump(nb_model, "models/naive_bayes.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ All models saved successfully in 'models/' folder")