import pandas as pd
import re
import nltk
import pickle
import os

import sys
sys.stdout.reconfigure(encoding='utf-8')

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"C:\Users\Admin\Desktop\Restaurant_Reviews.tsv",
    delimiter="\t",
    quoting=3
)

#  Dataset duplication (for experiment only)
df = pd.concat([df, df, df], ignore_index=True)

print("Dataset shape:", df.shape)


ps = PorterStemmer()
corpus = []

for review in df["Review"]:
    review = re.sub("[^a-zA-Z]", " ", review)
    review = review.lower().split()
    review = [ps.stem(word) for word in review]  # stopwords kept
    corpus.append(" ".join(review))


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)   # ❗ NO .toarray()
y = df.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

train_acc = rf_model.score(X_train, y_train)
test_acc = accuracy_score(y_test, y_pred)

bias = 1 - train_acc
variance = train_acc - test_acc

print("Train Accuracy:", train_acc)
print("Test Accuracy :", test_acc)
print("Bias          :", bias)
print("Variance      :", variance)


y_prob = rf_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print("Random Forest AUC:", auc)

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"RF (AUC = {auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.grid(True)
plt.show()


import pickle
import os

MODEL_DIR = r"C:\Users\Admin\Desktop\models"

# Load TF-IDF
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_loaded = pickle.load(f)

# Load Random Forest
with open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb") as f:
    rf_loaded = pickle.load(f)

print("Models loaded successfully")
