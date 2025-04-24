import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functions import *

nltk.download('punkt')
nltk.download('stopwords')

# documents for classification
documents = get_doc()  # Asus, Lenovo, Acer, Dell, Xiaomi

# Labels for the documents (for example purposes, replace with actual labels)
labels = ["Asus", "Lenovo", "Acer", "Acer", "Dell", "Xiaomi",]

# Tokenize the documents
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

# Preprocess all documents
processed_docs = [preprocess_text(doc) for doc in documents]

# Train Word2Vec model
model = Word2Vec(sentences=processed_docs, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")  # Save the model

# Convert documents to vectors
def document_to_vector(doc, model):
    word_vectors = [model.wv[word] for word in doc if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

document_vectors = np.array([document_to_vector(doc, model) for doc in processed_docs])

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(document_vectors, encoded_labels, test_size=0.2, random_state=42)

# Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")