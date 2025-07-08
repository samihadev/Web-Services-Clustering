from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import re
import numpy as np
import nltk
from collections import defaultdict

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_word_synonyms(word):
    """Get all synonyms for a word from WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def build_synonym_map(feature_names):
    """Create mapping of synonyms to a main word"""
    synonym_map = {}
    word_to_main = {}

    # Process words by length (longer words first)
    sorted_words = sorted(feature_names, key=len, reverse=True)

    for word in sorted_words:
        if word in word_to_main:
            continue  # Already processed

        synonyms = get_word_synonyms(word)
        synonyms.add(word)  # Include the word itself

        # Find if any synonym exists in our features
        for syn in synonyms:
            if syn in feature_names and syn != word:
                # If synonym exists, map to the shortest word
                main_word = min((word, syn), key=len)
                synonym_map[word] = main_word
                synonym_map[syn] = main_word
                word_to_main[word] = main_word
                word_to_main[syn] = main_word
                break

    return synonym_map

def process_tfidf_by_domain(texts, threshold=0.1, use_wordnet=True):
    """Process texts with TF-IDF and optionally merge synonyms using WordNet"""
    # Basic cleaning
    processed_texts = [
        re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
        for text in texts
    ]

    # Initial TF-IDF to get all features
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

    if use_wordnet:
        # Build synonym mapping if WordNet is enabled
        synonym_map = build_synonym_map(feature_names)

        # Merge synonyms in TF-IDF matrix
        merged_features = []
        merged_matrix = []
        main_words_used = set()

        for i, word in enumerate(feature_names):
            if word in synonym_map:
                main_word = synonym_map[word]
                if main_word not in main_words_used:
                    # Sum TF-IDF scores for all synonyms
                    synonym_indices = [j for j, w in enumerate(feature_names)
                                     if synonym_map.get(w, w) == main_word]
                    combined_score = tfidf_dense[:, synonym_indices].sum(axis=1)
                    merged_matrix.append(combined_score)
                    merged_features.append(main_word)
                    main_words_used.add(main_word)
            else:
                if word not in main_words_used:
                    merged_matrix.append(tfidf_dense[:, i])
                    merged_features.append(word)
                    main_words_used.add(word)

        # Convert to numpy array
        if merged_matrix:
            tfidf_merged = np.column_stack(merged_matrix)
        else:
            tfidf_merged = np.zeros((len(texts), 0))
    else:
        # Skip synonym merging if WordNet is disabled
        tfidf_merged = tfidf_dense
        merged_features = feature_names

    # Apply threshold
    tfidf_filtered = np.where(tfidf_merged >= threshold, tfidf_merged, 0)

    return tfidf_filtered, np.array(merged_features)