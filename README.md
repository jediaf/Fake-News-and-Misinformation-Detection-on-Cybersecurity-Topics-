# Fake-News-and-Misinformation-Detection-on-Cybersecurity-Topics
# 🛡️ Cybersecurity Fake News Detection

This project presents a hybrid system for detecting misinformation in cybersecurity-related news. It integrates a traditional machine learning pipeline (TF-IDF + Logistic Regression) with a transformer-based language model, accessible through a Flask web interface.

## 📁 Project Structure

├── app.py # Main Flask application
├── debug_models.py # Debugging and label-mapping utilities
├── retrain_model.py # Optional model retraining script
├── make_cyber_subset.py # Creates domain-specific cyber dataset
├── models/
│ ├── logreg_model_from_db.joblib
│ └── tfidf_vectorizer_from-db.joblib
├── templates/
│ └── index.html # Web interface
├── fake.csv # Raw fake news dataset
├── true.csv # Raw real news dataset
├── cyber.csv # Final cybersecurity dataset (balanced)
├── requirements.txt # Python dependencies
└── venv/ # Virtual environment (not uploaded to GitHub)

---

## 🎯 Objective

Cybersecurity misinformation, including exaggerated threat claims or fabricated breaches, poses real-world risks. This system enables **automated detection** of fake news articles in the cybersecurity domain using both statistical and contextual features.

---

## 🔧 Setup & Installation

1. Clone the repo:
```bash
git clone https://github.com/yourusername/cyber-fake-news-detector.git
cd cyber-fake-news-detector
#create a virtual enviroment:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:
pip install -r requirements.txt


🚀 Run the App
Make sure models/ directory contains the pre-trained .joblib files.
python app.py
Then open your browser and go to: http://127.0.0.1:5000
🧪 Models Used
TF-IDF + Logistic Regression
Fast, explainable, and trained on a balanced set of 3,048 cybersecurity articles.

(Optional) Transformer-based LLM
Not deployed in current app, but referenced in debug_models.py for analysis.

📊 Results
The TF-IDF + Logistic Regression model achieved:

Accuracy: 97%

F1-score: 0.97
This demonstrates robust performance even on nuanced cybersecurity content.

🛠️ Scripts
make_cyber_subset.py: Filters and balances cyber-related data.

retrain_model.py: Re-trains TF-IDF + Logistic Regression pipeline.

debug_models.py: Used to test LLM predictions and debug inconsistencies.
arXiv

👥 Contributors
Ceyda Arık

Barış Aygün


