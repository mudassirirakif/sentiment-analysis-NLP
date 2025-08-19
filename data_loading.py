
#This file take care of loading the reviews data from xml files.
#It will clean,filter outliers, encode the labels and shuffle data

import re
from pathlib import Path
from typing import List, Tuple
import random

def _review_parsing(file_path: str) -> List[str]:
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    reviews = []
    
    #get all the review tags
    for review_block in re.findall(r"<review>(.*?)</review>", text, re.S):
        match = re.search(r"<review_text>\s*(.*?)\s*</review_text>", review_block, re.S)
        if match:
            review_text = match.group(1).strip()
            reviews.append(review_text)
    return reviews


def _text_cleaning(text: str) -> str:
    no_html = re.sub(r"<[^>]+>", " ", text) #html tags remove
    no_punct = re.sub(r"[^\w\s']", " ", no_html) #html punctuation
    normalised = re.sub(r"\s+", " ", no_punct).strip().lower() #outlier remove
    return normalised


def load_reviews(
    positive_path: str,
    negative_path: str,
    min_words: int = 3,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:

    # calling review parsing function to remove the tags html
    pos_reviews = _review_parsing(positive_path)
    neg_reviews = _review_parsing(negative_path)

    texts: List[str] = []
    labels: List[int] = []

    # positive reviews cleaning
    for review in pos_reviews:
        cleaned = _text_cleaning(review)
        if len(cleaned.split()) >= min_words:
            texts.append(cleaned)
            labels.append(1)

    # negative reviews cleaning
    for review in neg_reviews:
        cleaned = _text_cleaning(review)
        if len(cleaned.split()) >= min_words:
            texts.append(cleaned)
            labels.append(0)

    # with random seed shuffle the dataset
    random.seed(seed)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    if combined:
        texts, labels = map(list, zip(*combined))
    else:
        texts, labels = [], []
    return texts, labels


__all__ = ["load_reviews"]