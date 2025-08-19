# main class that handles final predictions from training

import argparse
import joblib
import sys
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer  # optional, but helps
from sklearn.neural_network import MLPClassifier 

from typing import Tuple


def dependency_load(model_path: str, vectorizer_path: str):
    
    model = joblib.load(model_path)
    vectoriser = joblib.load(vectorizer_path)
    return model, vectoriser


def predict_sentiment(model, vectoriser, text: str) -> str:
    features = vectoriser.transform([text])
    pred = model.predict(features)[0]
    return "Positive review" if pred == 1 else "Negative review"


def user_interaction(model, vectoriser) -> None:
    print("Type a review to classify it. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("Review> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()  
            break
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        result = predict_sentiment(model, vectoriser, user_input)
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perform sentiment analysis on user input.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="sentiment_model.joblib",
        help="Path to the trained sentiment model",
    )
    parser.add_argument(
        "--vectorizer_path",
        type=str,
        default="vectorizer.joblib",
        help="Path to the fitted TF‑IDF vectoriser",
    )
    args = parser.parse_args()
    try:
        model, vectoriser = dependency_load(args.model_path, args.vectorizer_path)
    except FileNotFoundError as e:
        print(f"Error loading model or vectoriser: {e}")
        sys.exit(1)
    user_interaction(model, vectoriser)


if __name__ == "__main__":
    main()