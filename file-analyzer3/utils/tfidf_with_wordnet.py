from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import re
import numpy as np
import nltk
from collections import defaultdict
from sklearn.preprocessing import MaxAbsScaler

# Initialize NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def get_word_synonyms(word):
    """Get synonyms for a word using WordNet with error handling"""
    try:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace('_', ' '))
        return synonyms
    except:
        return set()


def build_synonym_map(feature_names):
    """Build a mapping of words to their main synonym forms"""
    synonym_map = {}
    word_to_main = {}

    # Sort words by length to process longer words first
    sorted_words = sorted(feature_names, key=len, reverse=True)

    for word in sorted_words:
        if word in word_to_main:
            continue

        synonyms = get_word_synonyms(word)
        synonyms.add(word)  # Include the word itself

        # Find if any synonym exists in our features
        for syn in synonyms:
            if syn in feature_names and syn != word:
                main_word = min((word, syn), key=len)  # Prefer shorter form
                synonym_map[word] = main_word
                synonym_map[syn] = main_word
                word_to_main[word] = main_word
                word_to_main[syn] = main_word
                break

    return synonym_map


def process_tfidf_by_domain(texts, threshold=0.1):
    """Process texts with TF-IDF and WordNet synonym merging"""
    # Basic text cleaning
    processed_texts = [re.sub(r'[^a-zA-Z\s]', '', text).lower().strip() for text in texts]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        stop_words='english',
        token_pattern=r'(?u)\b[a-zA-Z]+\b',
        max_df=0.85,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dense = tfidf_matrix.toarray()

    # Build synonym mapping and merge features
    synonym_map = build_synonym_map(feature_names)
    merged_features = []
    merged_matrix = []
    main_words_used = set()

    for i, word in enumerate(feature_names):
        if word in synonym_map:
            main_word = synonym_map[word]
            if main_word not in main_words_used:
                synonym_indices = [j for j, w in enumerate(feature_names)
                                   if synonym_map.get(w, w) == main_word]
                combined_score = tfidf_dense[:, synonym_indices].sum(axis=1)
                merged_matrix.append(combined_score)
                merged_features.append(main_word)
                main_words_used.add(main_word)
        elif word not in main_words_used:
            merged_matrix.append(tfidf_dense[:, i])
            merged_features.append(word)
            main_words_used.add(word)

    # Convert and normalize
    tfidf_merged = np.column_stack(merged_matrix) if merged_matrix else np.zeros((len(texts), 0))
    tfidf_merged = MaxAbsScaler().fit_transform(tfidf_merged)

    # Apply threshold
    tfidf_filtered = np.where(tfidf_merged >= threshold, tfidf_merged, 0)
    return tfidf_filtered, np.array(merged_features)