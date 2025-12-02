# Scripts/analysis.py

import pandas as pd
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import os

# Load models
sentiment_model = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
stopwords = nlp.Defaults.stop_words

# --- Sentiment Functions ---

def compute_sentiment(texts):
    results = sentiment_model(list(texts))
    labels = [r['label'].lower() for r in results]
    scores = [r['score'] for r in results]
    return labels, scores

def add_sentiment(df, text_col="review_text"):
    labels, scores = compute_sentiment(df[text_col])
    df["sentiment_label"] = labels
    df["sentiment_score"] = scores
    return df

def aggregate_sentiment(df, by=["bank_name", "rating"]):
    return df.groupby(by)[["sentiment_score"]].mean().reset_index()

# --- Thematic Functions ---

def preprocess_texts(texts):
    def clean(text):
        doc = nlp(text.lower())
        return " ".join([t.lemma_ for t in doc if t.is_alpha and t.text not in stopwords])
    return [clean(t) for t in texts]

def extract_keywords(texts, ngram_range=(1,2), top_k=10):
    tfidf = TfidfVectorizer(ngram_range=ngram_range, stop_words=list(stopwords))
    X = tfidf.fit_transform(texts)
    features = tfidf.get_feature_names_out()
    scores = X.toarray().sum(axis=0)
    top_idx = scores.argsort()[::-1][:top_k]
    return [features[i] for i in top_idx]

def cluster_keywords(keywords):
    themes = defaultdict(list)
    theme_map = {
        "Account Access Issues": ["login", "access", "password", "account"],
        "Transaction Performance": ["transfer", "transaction", "delay", "fail", "slow"],
        "User Interface & Experience": ["ui", "interface", "design", "navigation", "crash"],
        "Customer Support": ["support", "help", "service", "response"],
        "Feature Requests": ["feature", "add", "request", "option"]
    }
    for kw in keywords:
        assigned = False
        for theme, keys in theme_map.items():
            if any(k in kw for k in keys):
                themes[theme].append(kw)
                assigned = True
                break
        if not assigned:
            themes["Other"].append(kw)
    return dict(themes)

def assign_themes(df, text_col="review_text"):
    texts = preprocess_texts(df[text_col])
    keywords = extract_keywords(texts, ngram_range=(1,2), top_k=20)
    theme_dict = cluster_keywords(keywords)
    return theme_dict

# --- Full Pipeline Function ---

def run_analysis(input_csv, output_csv_sentiment=None, output_csv_themes=None):
    df = pd.read_csv(input_csv)
    
    # Sentiment
    df = add_sentiment(df, text_col="review_text")
    agg_sentiment = aggregate_sentiment(df)
    
    if output_csv_sentiment:
        os.makedirs(os.path.dirname(output_csv_sentiment), exist_ok=True)
        df.to_csv(output_csv_sentiment, index=False)
    
    # Thematic
    themes = assign_themes(df, text_col="review_text")
    if output_csv_themes:
        os.makedirs(os.path.dirname(output_csv_themes), exist_ok=True)
        pd.DataFrame.from_dict(themes, orient='index').to_csv(output_csv_themes)
    
    return df, agg_sentiment, themes
