import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="centered")

st.title("🎫 Support Ticket Classification & Prioritization")
st.markdown("Automatically classify customer support tickets and predict their priority level using Machine Learning.")

# Load Models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        cat_model = joblib.load('models/category_model.pkl')
        pri_model = joblib.load('models/priority_model.pkl')
        return vectorizer, cat_model, pri_model
    except Exception as e:
        return None, None, None

vectorizer, cat_model, pri_model = load_models()

if vectorizer is None or cat_model is None or pri_model is None:
    st.warning("Models are not loaded. Please ensure the notebook has been executed and models are saved in the 'models' directory.")
else:
    # Preprocessing Function
    stop_words = set(stopwords.words('english'))
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    # UI Inputs
    ticket_text = st.text_area("Enter Support Ticket Description:", height=150, placeholder="E.g., I cannot access my account after resetting my password.")

    if st.button("Predict"):
        if ticket_text.strip() == "":
            st.error("Please enter a ticket description.")
        else:
            with st.spinner("Analyzing ticket..."):
                # Preprocess
                clean_text = preprocess_text(ticket_text)
                X_input = vectorizer.transform([clean_text])
                
                # Predict
                cat_pred = cat_model.predict(X_input)[0]
                pri_pred = pri_model.predict(X_input)[0]
                
                # Confidences
                cat_probs = cat_model.predict_proba(X_input)[0]
                pri_probs = pri_model.predict_proba(X_input)[0]
                
                cat_conf = np.max(cat_probs) * 100
                pri_conf = np.max(pri_probs) * 100
                
            st.success("Prediction Complete!")
            
            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Predicted Category")
                st.info(f"**{cat_pred}**")
                st.caption(f"Confidence: {cat_conf:.2f}%")
                
            with col2:
                st.subheader("Predicted Priority")
                if pri_pred == 'High':
                    st.error(f"**{pri_pred}**")
                elif pri_pred == 'Medium':
                    st.warning(f"**{pri_pred}**")
                else:
                    st.success(f"**{pri_pred}**")
                st.caption(f"Confidence: {pri_conf:.2f}%")
                
            st.divider()
            st.markdown("### Confidence Scores Breakdown")
            
            # Show prob distribution
            cat_classes = cat_model.classes_
            cat_df = pd.DataFrame({"Category": cat_classes, "Probability": cat_probs})
            st.bar_chart(cat_df.set_index("Category"))
