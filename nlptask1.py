# ==========================================================
# STEP 0: Install & Import Libraries
# ==========================================================
import sys
import subprocess

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "nltk": "nltk"
}

for pkg, pip_name in packages.items():
    install_if_missing(pip_name)

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

print("✅ All libraries are installed and ready!")

# ==========================================================
# STEP 1: Load Dataset
# ==========================================================
# Replace with your dataset path

df = pd.read_csv("IMDB Dataset.csv")  # e.g., Kaggle IMDb reviews dataset
print("📊 Dataset Shape:", df.shape)
print(df.head())

# ==========================================================
# STEP 2: Text Cleaning
# ==========================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned'] = df['review'].apply(clean_text)

# Encode labels (1 = positive, 0 = negative)
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# ==========================================================
# STEP 3: Vectorization (TF-IDF)
# ==========================================================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned']).toarray()
y = df['label']

# ==========================================================
# STEP 4: Train-Test Split
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# STEP 5: Logistic Regression
# ==========================================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, y_pred_lr)
print(f"\n📈 Logistic Regression Accuracy: {lr_acc:.4f}")
print(classification_report(y_test, y_pred_lr))

# ==========================================================
# STEP 6: Naive Bayes
# ==========================================================
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, y_pred_nb)
print(f"\n📈 Naive Bayes Accuracy: {nb_acc:.4f}")
print(classification_report(y_test, y_pred_nb))

# ==========================================================
# STEP 7: Compare Models
# ==========================================================
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [lr_acc, nb_acc]
})
print("\n📊 Model Comparison:\n", results)

# ==========================================================
# STEP 8: Visualize Most Frequent Positive & Negative Words
# ==========================================================
positive_words = ' '.join(df[df['label'] == 1]['cleaned']).split()
negative_words = ' '.join(df[df['label'] == 0]['cleaned']).split()

pos_freq = pd.Series(positive_words).value_counts().head(20)
neg_freq = pd.Series(negative_words).value_counts().head(20)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.barplot(x=pos_freq.values, y=pos_freq.index, color="green")
plt.title("Top Positive Words")

plt.subplot(1,2,2)
sns.barplot(x=neg_freq.values, y=neg_freq.index, color="red")
plt.title("Top Negative Words")

plt.tight_layout()
plt.show()
