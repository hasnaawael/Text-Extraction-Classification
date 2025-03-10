# classify_documents.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def classify_documents(extracted_data, topics):
    """Classifies documents based on extracted topics."""
    classified_data = []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    text_matrix = vectorizer.fit_transform([doc["text"] for doc in extracted_data])

    lda = LatentDirichletAllocation(n_components=len(topics), random_state=42)
    topic_distributions = lda.fit_transform(text_matrix)

    for i, doc in enumerate(extracted_data):
        topic_idx = np.argmax(topic_distributions[i])
        predicted_category = topics.get(f"Topic_{topic_idx+1}", "Unknown")

        classified_data.append({
            "file_name": doc["file_name"],
            "text": doc["text"],
            "category": predicted_category
        })

    return classified_data
