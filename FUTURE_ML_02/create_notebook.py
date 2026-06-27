import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Support Ticket Classification & Prioritization\nThis notebook covers Data Preprocessing, EDA, Model Training, and Evaluation."),
        
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
import string
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report"""),

        nbf.v4.new_markdown_cell("## 1. Data Preprocessing"),
        nbf.v4.new_code_cell("""# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv('../dataset/support_tickets.csv')
print("Shape before cleaning:", df.shape)

# Handle missing values
df = df.dropna(subset=['Ticket Description'])

# Remove duplicates
df = df.drop_duplicates(subset=['Ticket Description'])
print("Shape after cleaning:", df.shape)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['Clean_Description'] = df['Ticket Description'].apply(preprocess_text)
df.head()"""),

        nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis"),
        nbf.v4.new_code_cell("""# Set visualization style
sns.set(style="whitegrid")
os.makedirs('../images', exist_ok=True)

# 1. Ticket Category Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Category', palette='viridis', order=df['Category'].value_counts().index)
plt.title("Ticket Category Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../images/category_distribution.png')
plt.show()

# 2. Priority Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Priority Level', palette='magma', order=['High', 'Medium', 'Low'])
plt.title("Priority Level Distribution")
plt.tight_layout()
plt.savefig('../images/priority_distribution.png')
plt.show()

# 3. Word Cloud
all_words = " ".join(df['Clean_Description'])
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(all_words)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Ticket Descriptions")
plt.tight_layout()
plt.savefig('../images/wordcloud.png')
plt.show()"""),

        nbf.v4.new_markdown_cell("## 3. Feature Engineering & Modeling for Category Classification"),
        nbf.v4.new_code_cell("""# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Clean_Description'])
y_category = df['Category']

# Split Data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_category, test_size=0.2, random_state=42)

# Models
models_cat = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_acc = 0
best_model_cat = None

for name, model in models_cat.items():
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model_cat = model

print("Best Category Model:", best_model_cat)

# Evaluation
y_pred_c = best_model_cat.predict(X_test_c)
print(classification_report(y_test_c, y_pred_c))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model_cat.classes_, yticklabels=best_model_cat.classes_)
plt.title("Confusion Matrix - Category Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('../images/confusion_matrix.png')
plt.show()"""),

        nbf.v4.new_markdown_cell("## 4. Feature Engineering & Modeling for Priority Prediction"),
        nbf.v4.new_code_cell("""y_priority = df['Priority Level']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.2, random_state=42)

models_pri = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_acc_p = 0
best_model_pri = None

for name, model in models_pri.items():
    model.fit(X_train_p, y_train_p)
    preds = model.predict(X_test_p)
    acc = accuracy_score(y_test_p, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_acc_p:
        best_acc_p = acc
        best_model_pri = model

print("Best Priority Model:", best_model_pri)
y_pred_p = best_model_pri.predict(X_test_p)
print(classification_report(y_test_p, y_pred_p))"""),

        nbf.v4.new_markdown_cell("## 5. Export Models"),
        nbf.v4.new_code_cell("""os.makedirs('../models', exist_ok=True)
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
joblib.dump(best_model_cat, '../models/category_model.pkl')
joblib.dump(best_model_pri, '../models/priority_model.pkl')
print("Models exported successfully!")""")
    ]

    nb['cells'] = cells
    with open('notebook/Support_Ticket_Classification.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("Notebook created successfully!")

if __name__ == "__main__":
    create_notebook()
