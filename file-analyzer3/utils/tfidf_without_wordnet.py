from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from sklearn.preprocessing import MaxAbsScaler


def process_tfidf_by_domain(texts, threshold=0.1):
    # Basic cleaning
    processed_texts = [re.sub(r'[^a-zA-Z\s]', '', text).lower().strip() for text in texts]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z]+\b',
        max_df=0.85,
        min_df=2,
        max_features=1000
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dense = tfidf_matrix.toarray()

    # Normalization - Key fix
    tfidf_dense = MaxAbsScaler().fit_transform(tfidf_dense)

    # Apply threshold
    tfidf_filtered = np.where(tfidf_dense >= threshold, tfidf_dense, 0)
    return tfidf_filtered, feature_names