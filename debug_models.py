# debug_models.py
import joblib, pathlib, traceback
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

TXT_REAL = "Microsoft issues emergency patch for critical Windows zero-day."
TXT_FAKE = "Alien hackers demand Bitcoin ransom after taking over NASA satellites."

# ---------- TF-IDF baseline ----------
base = pathlib.Path("models")
vec  = joblib.load(base/"tfidf_vectorizer_from_db.joblib")
clf  = joblib.load(base/"logreg_model_from_db.joblib")

for txt in (TXT_REAL, TXT_FAKE):
    pred = clf.predict(vec.transform([txt]))[0]
    print(f"TF-IDF → {txt[:40]}…  ==>  {pred}")

# ---------- LLM ----------
model_id = "mrm8488/bert-tiny-finetuned-fake-news-detection"
print("\nLoading HF model once …")
try:
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=False)
    hf  = pipeline("text-classification", model=mdl, tokenizer=tok, top_k=1)
    for txt in (TXT_REAL, TXT_FAKE):
        lbl = hf(txt, truncation=True)[0]["label"].upper()
        print(f"LLM      → {txt[:40]}…  ==>  {lbl}")
except Exception:
    traceback.print_exc()
