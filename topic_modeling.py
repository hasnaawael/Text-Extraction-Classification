# topic_modeling.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_topics(texts, num_topics=5, num_words=10):
    """Extracts common topics from text using LDA."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    text_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(text_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_idx, topic in enumerate(lda.components_):
        topics[f"Topic_{topic_idx+1}"] = " ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]])

    return topics
