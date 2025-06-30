# app.py  – TF-IDF baseline + BERT-tiny fake-news LLM
from flask import Flask, render_template, request
import joblib, pathlib
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# ── load the fine-tuned fake-news model once ─────────────────────────
MODEL_ID = "mrm8488/bert-tiny-finetuned-fake-news-detection"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
hf_clf = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection",
    tokenizer="mrm8488/bert-tiny-finetuned-fake-news-detection"
)   # ← no top_k



def llm_predict(text: str) -> str:
    try:
        label = hf_clf(text, truncation=True, max_length=512)[0]["label"]
        return "Fake" if label == "LABEL_0" else "Real"
    except Exception as e:
        print("LLM error:", e)
        return "Error"




# ── load TF-IDF + Logistic Regression baseline ───────────────────────
BASE = pathlib.Path(__file__).parent
clf = joblib.load(BASE / "models/logreg_model_from_db.joblib")
vec = joblib.load(BASE / "models/tfidf_vectorizer_from_db.joblib")

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)
LABEL_MAP = {
    "bs": "Fake",
    "bias": "Fake",
    "satire": "Fake",
    "conspiracy": "Fake",
    "junksci": "Fake",
    "hate": "Fake",
    "fake": "Fake",
    "true": "Real",
    "state": "Real",
    "real": "Real"
}
@app.route("/", methods=["GET", "POST"])
def home():
    text = tfidf_pred = llm_pred = None
    if request.method == "POST":
        text = request.form.get("news", "")
        if text.strip():
            raw = clf.predict(vec.transform([text]))[0]
            tfidf_pred = LABEL_MAP.get(raw.lower(), raw.title())            
            llm_pred   = llm_predict(text)
    return render_template("index.html", text=text or "", tfidf=tfidf_pred, llm=llm_pred)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
