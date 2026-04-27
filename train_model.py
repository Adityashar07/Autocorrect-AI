import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "words_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")

# ── Read CSV (auto-detect separator) ──────────────────────────────────────────
for sep in [',', '\t', ';']:
    try:
        df = pd.read_csv(DATA_PATH, sep=sep, engine='python')
        if 'corrected_form' in df.columns:
            print(f"CSV loaded with separator: '{sep}'")
            break
    except Exception:
        continue
else:
    raise ValueError("Could not read CSV or 'corrected_form' column not found. "
                     "Check your dataset/words_dataset.csv file.")

print("Columns found:", df.columns.tolist())
print(f"Rows in dataset: {len(df)}")

# ── Build training data ────────────────────────────────────────────────────────
rows = []
for _, r in df.iterrows():
    correct = str(r['corrected_form']).strip().lower()
    if not correct or correct == 'nan':
        continue
    # Add the correct word itself
    rows.append({"misspelled": correct, "correct": correct})
    # Add all misspelling variants
    misspellings = str(r.get('misspellings', '')).strip()
    for typo in misspellings.split("|"):
        typo = typo.strip().lower()
        if typo and typo != 'nan':
            rows.append({"misspelled": typo, "correct": correct})
    # Also add the 'word' column entry as a misspelling
    word_col = str(r.get('word', '')).strip().lower()
    if word_col and word_col != 'nan' and word_col != correct:
        rows.append({"misspelled": word_col, "correct": correct})

data = pd.DataFrame(rows).drop_duplicates()
print(f"Total training samples: {len(data)}")

X = data["misspelled"].astype(str)
y = data["correct"].astype(str)

# ── Train model ────────────────────────────────────────────────────────────────
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000, C=5, solver="lbfgs")),
])

model.fit(X, y)
print("Model trained successfully!")

# ── Save model ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print("Done! Now run: python app.py")