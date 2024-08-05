# Download dataset : !gdown 1jh2p2DlaWsDo_vEWIcTrNh3mUuXd-cw6
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def create_data():
    return pd.read_csv("./vi_text_retrieval.csv")


def process_text_data():
    vi_data_df = create_data()
    context = vi_data_df['text']
    context = [doc.lower() for doc in context]

    tfidf_vectorizer = TfidfVectorizer()
    context_embedded = tfidf_vectorizer.fit_transform(context)
    value = context_embedded.toarray()[7][0]
    return value


def tfidf_search(question, tfidf_vectorizer, context_embedded, top_d=5):
    question = question.lower()

    query_embedded = tfidf_vectorizer.transform([question])

    cosine_scores = cosine_similarity(
        query_embedded, context_embedded).flatten()

    results = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)

    return results


def corr_search(question, tfidf_vectorizer, context_embedded, top_d=5):
    question = question.lower()

    query_embedded = tfidf_vectorizer.transform([question])

    corr_scores = np.corrcoef(query_embedded.toarray(),
                              context_embedded.toarray())

    corr_scores = corr_scores[0][1:]

    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)

    return results
