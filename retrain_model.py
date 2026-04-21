import os
import re
import string

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = os.path.dirname(os.path.abspath(__file__))


def wordpre(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def to_article_label(df: pd.DataFrame, article_col: str, label_col: str) -> pd.DataFrame:
    out = df[[article_col, label_col]].copy()
    out.columns = ["Article", "label"]
    out = out.dropna(subset=["Article", "label"])
    out["Article"] = out["Article"].astype(str).apply(wordpre)
    return out


def build_dataset() -> pd.DataFrame:
    d1 = pd.read_csv(os.path.join(ROOT, "Data_set", "news", "news.csv"))
    d1["Article"] = d1["title"].fillna("") + " " + d1["text"].fillna("")
    d1["label"] = d1["label"].replace({"REAL": 1, "FAKE": 0})
    d1 = to_article_label(d1, "Article", "label")

    d2_true = pd.read_csv(os.path.join(ROOT, "Dataset", "True.csv"))
    d2_fake = pd.read_csv(os.path.join(ROOT, "Dataset", "Fake.csv"))
    d2_true["label"] = 1
    d2_fake["label"] = 0
    d2 = pd.concat([d2_true, d2_fake], ignore_index=True)
    d2["Article"] = d2["title"].fillna("") + " " + d2["text"].fillna("")
    d2 = to_article_label(d2, "Article", "label")

    d3_real = pd.read_csv(os.path.join(ROOT, "politifact", "politifact_real.csv"))
    d3_fake = pd.read_csv(os.path.join(ROOT, "politifact", "politifact_fake.csv"))
    d3_real["label"] = 1
    d3_fake["label"] = 0
    d3 = pd.concat([d3_real, d3_fake], ignore_index=True)
    d3["Article"] = d3["title"].fillna("")
    d3 = to_article_label(d3, "Article", "label")

    d4 = pd.read_csv(os.path.join(ROOT, "Data-set", "train.csv"))
    d4["Article"] = d4["title"].fillna("") + " " + d4["text"].fillna("")
    d4 = to_article_label(d4, "Article", "label")

    d5 = pd.read_csv(os.path.join(ROOT, "data.csv"))
    d5["Article"] = d5["Headline"].fillna("") + " " + d5["Body"].fillna("")
    d5["label"] = d5["Label"]
    d5 = to_article_label(d5, "Article", "label")

    dataset = pd.concat([d1, d2, d3, d4, d5], ignore_index=True)
    dataset["label"] = pd.to_numeric(dataset["label"], errors="coerce")
    dataset = dataset.dropna(subset=["label", "Article"])
    dataset["label"] = dataset["label"].astype(int)
    return dataset.sample(frac=1.0, random_state=2020).reset_index(drop=True)


def train_and_save(model_path: str) -> None:
    dataset = build_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["Article"],
        dataset["label"],
        test_size=0.2,
        random_state=2020,
    )

    model = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            (
                "model",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, model_path)

    print(f"Records used: {len(dataset)}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    output_path = os.path.join(ROOT, "model.pkl")
    train_and_save(output_path)
