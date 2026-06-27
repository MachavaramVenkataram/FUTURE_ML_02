# Project Report: Support Ticket Classification & Prioritization

## 1. Introduction
Customer support is a critical component of any business. Handling thousands of support tickets manually is time-consuming, prone to error, and often results in delayed responses for urgent issues. This project introduces a Machine Learning and Natural Language Processing (NLP) based system that automatically categorizes support tickets and predicts their priority level, enabling support teams to resolve issues efficiently.

## 2. Problem Statement
Given a textual description of a support ticket, the system must:
1. Classify the ticket into one of the predefined **Categories** (Billing Issues, Technical Issues, Account Access, Product Inquiry, General Query).
2. Predict the **Priority Level** (High, Medium, Low) to help support teams handle critical cases first.

## 3. Dataset Description
A synthetic, realistic dataset (`support_tickets.csv`) was generated comprising over 2,000 records.
- **Features**:
  - `Ticket ID`: Unique identifier for the ticket.
  - `Ticket Description`: The text content of the support request.
  - `Date Created`: The timestamp when the ticket was generated.
- **Target Variables**:
  - `Category`: The department/topic the ticket belongs to.
  - `Priority Level`: The urgency of the ticket.

## 4. Text Preprocessing
Raw text data contains noise that can negatively impact ML models. The following NLP preprocessing steps were applied:
- **Lowercasing**: Converted all text to lowercase to ensure uniformity.
- **Punctuation Removal**: Removed special characters and punctuation marks.
- **Tokenization**: Split sentences into individual words (tokens) using `NLTK`.
- **Stopword Removal**: Removed common English words (e.g., "is", "the", "and") that do not contribute to predictive meaning.

## 5. Feature Extraction
To convert text into numerical formats suitable for machine learning, **Term Frequency-Inverse Document Frequency (TF-IDF)** was utilized.
- TF-IDF weighs the frequency of a word against its rarity across all tickets, effectively highlighting important keywords (e.g., "charged", "crash", "password").
- A maximum of 5,000 features were extracted to maintain computational efficiency while capturing the most relevant vocabulary.

## 6. Model Development
Two distinct sets of models were developed for Category Classification and Priority Prediction. The dataset was split into 80% training and 20% testing sets.

The following algorithms were evaluated:
- **Logistic Regression**: A robust baseline for text classification.
- **Naive Bayes (MultinomialNB)**: Highly effective for word-count and frequency-based text classification.
- **Random Forest Classifier**: An ensemble method capable of capturing complex, non-linear relationships.

## 7. Evaluation Metrics
Models were evaluated using the following metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of Precision and Recall.
- **Confusion Matrix**: Visual representation of true vs. predicted classifications.

## 8. Results
- **Category Classification**: Logistic Regression and Random Forest performed exceptionally well, achieving over 98% accuracy on the synthetic dataset, due to distinct keyword separation across categories.
- **Priority Prediction**: Random Forest provided strong performance in classifying High, Medium, and Low priorities, effectively capturing phrases indicating urgency like "ASAP", "urgent", or "hacked".

## 9. Visualizations
Detailed visualizations generated during Exploratory Data Analysis include:
1. **Ticket Category Distribution**: Displaying the volume of tickets per department.
2. **Priority Distribution**: Showing the split between High, Medium, and Low urgency tickets.
3. **Word Cloud**: Visualizing the most prominent terms across all ticket descriptions.
4. **Confusion Matrix**: Highlighting the model's prediction accuracy per class.

*(All generated visual plots can be found in the `images/` directory)*

## 10. Business Benefits
1. **Reduced Response Time**: High-priority tickets are immediately flagged and routed to the correct department.
2. **Operational Efficiency**: Eliminates the need for manual triaging, freeing up agents to focus on resolving issues.
3. **Cost Savings**: Automated routing reduces the overhead associated with Level 1 support triage.
4. **Data-Driven Insights**: Provides management with clear visibility into the most common customer pain points.

## 11. Conclusion
The "Support Ticket Classification & Prioritization" project successfully demonstrates how NLP and Machine Learning can automate the customer service workflow. By employing TF-IDF and traditional ML models, the system can reliably interpret unstructured text and route tickets with high accuracy.

## 12. Future Scope
- **Advanced NLP Models**: Implementing Transformer-based models like BERT or RoBERTa for deeper semantic understanding.
- **Continuous Learning**: Building a feedback loop where agent corrections automatically retrain the model.
- **Multi-lingual Support**: Expanding the preprocessing pipeline to handle non-English tickets.
- **Deployment**: Packaging the model as a REST API using FastAPI/Flask and deploying to Cloud platforms (AWS/GCP/Azure).
