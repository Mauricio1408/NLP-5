from collections import Counter
from functions import *

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

# Build vocabulary
vocabulary = set(word for doc in tokenized_docs for word in doc)

# Compute TF, IDF, and TF-IDF vectors
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]
idf = compute_idf(tokenized_docs, vocabulary)
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

# Compare all document pairs
most_similar_pair = (0, 0)
highest_similarity = -1

for i in range(len(tfidf_vectors)):
    for j in range(i + 1, len(tfidf_vectors)):
        similarity = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j], vocabulary)
        print(f"Cosine Similarity between Document {i+1} and Document {j+1}: {similarity:.4f}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_pair = (i, j)

# Output the most similar pair
doc1, doc2 = most_similar_pair
print(f"\nMost similar documents: Document {doc1 + 1} and Document {doc2 + 1}")
print(f"Highest Cosine Similarity: {highest_similarity:.4f}")

