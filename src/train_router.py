from pathlib import Path
import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_FILE = BASE_DIR / "data" / "router_training.csv"
MODEL_FILE = BASE_DIR / "data" / "router_trained.joblib"


def load_training_data():
    prompts = []
    labels = []

    with open(TRAINING_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            prompts.append(row["prompt"])
            labels.append(row["label"])

    return prompts, labels


def train():
    prompts, labels = load_training_data()

    classifier = Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    classifier.fit(prompts, labels)

    joblib.dump(classifier, MODEL_FILE)

    print(f"Training completato.")
    print(f"Modello salvato in: {MODEL_FILE}")


if __name__ == "__main__":
    train()