
# this file use to train and save the models

import argparse
import joblib
from typing import Tuple, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_loading import load_reviews
from model import prepare_model

import os
import glob
import random

def train_and_evaluate(positive_path: Optional[str],negative_path: Optional[str],model_output: str,vectorizer_output: str,min_words: int = 3,seed: int = 42,data_root: Optional[str] = None,) -> None:
    print("Loading and preprocessing reviews...")
    texts = []
    labels = []

    if data_root:
        # get positive and negative reviews
        pos_files = sorted(
            glob.glob(os.path.join(data_root, "**", "positive.review"), recursive=True)
        )
        neg_files = sorted(
            glob.glob(os.path.join(data_root, "**", "negative.review"), recursive=True)
        )
        if not pos_files or not neg_files:
            raise FileNotFoundError(
                f"No positive.review or negative.review files found under {data_root}"
            )
        # additional check
        if len(pos_files) != len(neg_files):
            print("both reviews not equal so class imbalance can occur")
        all_files = list(zip(pos_files, neg_files))
        print(f"Found {len(all_files)} domain(s) with review files.")
        for pos_fp, neg_fp in all_files:
            domain_texts, domain_labels = load_reviews(
                positive_path=pos_fp,
                negative_path=neg_fp,
                min_words=min_words,
                seed=seed,
            )
            texts.extend(domain_texts)
            labels.extend(domain_labels)
        random.seed(seed)
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = map(list, zip(*combined))
    else:
        if not (positive_path and negative_path):
            raise ValueError("invalid path")
        texts, labels = load_reviews(
            positive_path=positive_path,
            negative_path=negative_path,
            min_words=min_words,
            seed=seed,
        )

    print(
        f"Loaded {len(texts)} reviews (positive: {sum(labels)}, negative: {len(labels) - sum(labels)})"
    )

    print("spilting and vectorizing the data...")
    vectoriser, X_train, X_val, X_test, y_train, y_val, y_test = df_prepare(
        texts, labels, test_size=0.2, val_size=0.1, seed=seed
    )
    print("model building...")
    clf = prepare_model(random_state=seed)
    # fit the model
    print("Training model...")
    clf.fit(X_train, y_train)

    # check metrics like accuracy
    val_accuracy = clf.score(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    print("train+validation data retraining")
    X_train_full = vectoriser.transform(texts[: len(texts)])
    # new classifier
    final_clf = prepare_model(random_state=seed)
    final_clf.fit(X_train_full, labels)
    
    test_predictions = final_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Classification report:\n")
    print(classification_report(y_test, test_predictions, target_names=["Negative", "Positive"]))

    # save models
    print(f"Saving model to {model_output} and vectoriser to {vectorizer_output}...")
    joblib.dump(final_clf, model_output)
    joblib.dump(vectoriser, vectorizer_output)
    print("Training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model for NLP problem i.e sentiment analysis")
    parser.add_argument(
        "--positive_path",
        type=str,
        default=None,
        help="Path to the positive reviews file (ignored when --data_root is set)",
    )
    parser.add_argument(
        "--negative_path",
        type=str,
        default=None,
        help="Path to the negative reviews file (ignored when --data_root is set)",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="sentiment_model.joblib",
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--vectorizer_output",
        type=str,
        default="vectorizer.joblib",
        help="Where to save the fitted TF‑IDF vectoriser",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=3,
        help="Minimum number of words required to keep a review",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root path with all reviews of all categories in the folder"
    )
    args = parser.parse_args()

    train_and_evaluate(
        positive_path=args.positive_path,
        negative_path=args.negative_path,
        model_output=args.model_output,
        vectorizer_output=args.vectorizer_output,
        min_words=args.min_words,
        seed=args.seed,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()