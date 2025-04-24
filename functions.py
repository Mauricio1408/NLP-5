# Accessing Wikipedia
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize

from collections import Counter
import wikipedia
import re
import math


# Computer Term Frequency
# Given a list of tokens and a vocabulary, compute the term frequency for each term in the vocabulary.
def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }


# Compute the Inverse Document Frequency
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        # Count the number of documents containing the term
        df = sum(term in doc for doc in tokenized_docs)
        # Compute IDF using the formula: idf(t) = lo(N / df(t))
        idf_dict[term] = math.log(N / (df or 1))
    return idf_dict


# Computer TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    # Compute the TF-IDF score for each term in the vocabulary
    # using the formula: tf-idf(t, d) = tf(t, d) * idf(t)
    return { term: tf_vector[term] * idf[term] for term in vocab}


# Calculate the Cosine Similarity
def cosine_similarity(vec1, vec2, vocab):
    # Get the dot product of the two vectors
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1Length = math.sqrt(sum(vec1[term]**2 for term in vocab))
    vec2Length = math.sqrt(sum(vec2[term]**2 for term in vocab))

    if vec1Length == 0 or vec2Length == 0:
        return 0.0

    return dot_product / (vec1Length * vec2Length)


# get the wiki content of a given topic
def get_wiki_content(topic):
    return wikipedia.page(topic).content[:1000]


# Get the content of the Wikipedia pages for the specified topics
# and store them in a list. The content is limited to the first 1000 characters for simplicity.
def get_doc():
    asus = get_wiki_content("Asus Tech")
    lenovo = get_wiki_content("Lenovo")
    acer = get_wiki_content("Acer")
    dell = get_wiki_content("Dell")
    xiaomi = get_wiki_content("Xiaomi Tech")
    
    document = [
        asus,
        lenovo,
        acer,
        dell,
        xiaomi    
    ]

    return document