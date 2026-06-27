# Support Ticket Classification & Prioritization

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

## 📌 Project Overview
This project is an NLP-based machine learning system designed to automatically classify customer support tickets into predefined categories and predict their priority level (High, Medium, Low). The goal is to streamline customer support operations, improve response times, and identify critical issues automatically.

### Key Features:
- **Text Classification**: Uses TF-IDF Vectorization to process unstructured ticket descriptions.
- **Category Prediction**: Classifies tickets into categories like *Billing Issues, Technical Issues, Account Access, Product Inquiry*, and *General Query*.
- **Priority Prediction**: Predicts ticket priority (*High, Medium, Low*) based on urgency and context.
- **Interactive Web App**: A user-friendly Streamlit web application to test predictions in real-time.
- **Visual Analytics**: Includes visualizations for ticket distribution, priority breakdown, and common keywords.

---

## 🚀 Installation Steps

1. **Clone the repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/yourusername/Support-Ticket-Classification.git
   cd Support-Ticket-Classification
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data (done automatically by scripts, but can be done manually)**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## 💻 Usage Instructions

### 1. Generating the Dataset
To create the synthetic dataset (if not already present), run:
```bash
python generate_dataset.py
```
This generates `dataset/support_tickets.csv`.

### 2. Training the Models
To train the models and generate visualizations, execute the Jupyter Notebook:
```bash
jupyter notebook notebook/Support_Ticket_Classification.ipynb
```
Run all cells. The trained models will be saved in the `models/` directory, and evaluation charts will be saved in `images/`.

### 3. Running the Web App
Start the Streamlit application:
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser.

---

## 📊 Sample Predictions

| Ticket Description | Predicted Category | Predicted Priority |
|--------------------|--------------------|--------------------|
| "I cannot access my account after resetting my password." | Account Access | High |
| "I would like information about your premium subscription plan." | Product Inquiry | Low |
| "The app crashes every time I try to open the dashboard." | Technical Issues | High |

---

## 📈 Results
- **Category Model (Random Forest / Logistic Regression)**: Achieved ~99%+ accuracy on synthetic data.
- **Priority Model (Random Forest / Logistic Regression)**: Achieved ~95%+ accuracy on synthetic data.
*(Actual results may vary based on random seed and dataset generation)*

See the `images/` directory for detailed visualizations such as the Confusion Matrix, Word Cloud, and Category Distributions.

---

## 📁 Project Structure
```text
Support-Ticket-Classification/
│
├── dataset/
│   └── support_tickets.csv         # Generated dataset
│
├── notebook/
│   └── Support_Ticket_Classification.ipynb # EDA and Model Training
│
├── images/                         # Saved visualizations
│   ├── category_distribution.png
│   ├── priority_distribution.png
│   ├── confusion_matrix.png
│   └── wordcloud.png
│
├── report/
│   └── Project_Report.md           # Detailed Project Report
│
├── models/                         # Saved ML models (.pkl)
│   ├── tfidf_vectorizer.pkl
│   ├── category_model.pkl
│   └── priority_model.pkl
│
├── app.py                          # Streamlit web application
├── generate_dataset.py             # Script to generate data
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview and instructions
```
