#!/usr/bin/env python3
import argparse
import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack


IGNORE_WORDS = [
    "aaaaaaaa"   # keep if you want to ignore garbage tokens
]

def normalize_text(t: str) -> str:
    t = str(t).lower().strip()
    for w in IGNORE_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s']", "", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser(description="Train AMD text classifier")
    ap.add_argument("--csv", required=True, help="CSV with columns: text,label")
    ap.add_argument("--out", required=True, help="Output joblib path")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading dataset:", args.csv)
    df = pd.read_csv(args.csv)

    assert "text" in df.columns and "label" in df.columns, \
        "CSV must contain columns: text,label"


    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].apply(normalize_text)
    df = df[df["text"].str.len() > 0]

    label_map = {"HUMAN": 0, "MACHINE": 1}
    df["y"] = df["label"].map(label_map)

    print("Label distribution:")
    print(df["label"].value_counts())

    X = df["text"].values
    y = df["y"].values

   
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

   
    print("Fitting WORD TF-IDF...")
    word_v = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=12000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=True
    )

    Xtr_word = word_v.fit_transform(X_train)
    Xte_word = word_v.transform(X_test)


    print("Fitting CHAR TF-IDF...")
    char_v = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=6000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    Xtr_char = char_v.fit_transform(X_train)
    Xte_char = char_v.transform(X_test)

    
    Xtr = hstack([Xtr_word, Xtr_char])
    Xte = hstack([Xte_word, Xte_char])

    print("Features:")
    print(" Word:", Xtr_word.shape[1])
    print(" Char:", Xtr_char.shape[1])
    print(" Total:", Xtr.shape[1])


    print("Training Logistic Regression...")
    base_clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed
    )

   
    clf = CalibratedClassifierCV(
        base_clf,
        method="sigmoid",
        cv=5,
        n_jobs=-1
    )

    clf.fit(Xtr, y_train)

  
    y_pred = clf.predict(Xte)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["HUMAN", "MACHINE"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

   
    bundle = {
        "clf": clf,
        "word_v": word_v,
        "char_v": char_v,
        "vectorizer": word_v,  # backward compatibility
        "label_map": label_map
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)

    print("\n Model saved to:", args.out)

if __name__ == "__main__":
    main()
