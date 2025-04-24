from collections import Counter
from functions import get_doc, compute_tf, compute_idf, compute_tfidf

import re
from nltk.tokenize import word_tokenize
import pandas as pd

# Fetch documents from Wikipedia
documents = get_doc()  # Asus, Lenovo, Acer, Dell, Xiaomi

print("Documents:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc[:200]}...\n")  # Show a snippet

# Tokenize and lowercase
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [re.sub(r'\W+', '', t) for t in tokens if re.sub(r'\W+', '', t)]

tokenized_docs = [preprocess(doc) for doc in documents]

# Create vocabulary (set of all unique words)
vocabulary = set(term for doc in tokenized_docs for term in doc)

# Compute TF vectors
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

print("\nTerm Frequency Vectors:")
for i, tf_vector in enumerate(tf_vectors):
    print(f"Document {i+1}: {dict(list(tf_vector.items())[:10])} ...")  # preview top 10 terms

# Compute IDF
idf = compute_idf(tokenized_docs, vocabulary)

print("\nInverse Document Frequency:")
for term in list(idf)[:10]:  # preview top 10 terms
    print(f"{term}: {idf[term]}")

# Compute TF-IDF vectors
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

print("\nTF-IDF Vectors:")
for i, tfidf_vector in enumerate(tfidf_vectors):
    print(f"Document {i+1}: {dict(list(tfidf_vector.items())[:10])} ...")  # preview


# Term Frequency (TF) Observations:
    # Common word “is” appears in all documents, giving it a non-discriminative value (it’s a stopword-level term).
    # Specific terms like “obrien”, “conan”, “oxen”, “british” appear in very few documents, often with non-zero frequency in only one.
    # TF highlights how often terms appear within each document, but doesn’t indicate importance across the corpus.


# Inverse Document Frequency (IDF) Insights:
    # "is" has an IDF of 0, meaning it appears in every document → carries no distinguishing power.
    # Other words (e.g. “started”, “obrien”, “conan”, “oxen”) have high IDF (~1.61), meaning they appear in only one document, making them highly distinctive.
      
# TF-IDF Observations:
    # TF-IDF zeros out words like “is” despite high raw frequency — showing its irrelevance for document distinction.
    # Document 2 (Jay Leno) has high TF-IDF scores for “obrien”, “conan”, and “revival” → terms strongly associated with that document only.
    # Document 3 (acre) is uniquely identified by “oxen”, “british”, and “man”.
    # Document 4 (mathematics) is differentiated by “corresponding”.
    # Document 5 has mostly 0 TF-IDF values for the sampled words → suggests these words are not relevant to its core topic or are absent.

# This means that:
    # TF-IDF effectively filters out generic/common words while emphasizing unique or defining vocabulary for each document.
    # The most distinctive documents based on sampled TF-IDF terms are Documents 2, 3, and 4 — likely due to focused and topic-specific language.
    # Document 5 may need further analysis or different terms to extract distinguishing keywords.