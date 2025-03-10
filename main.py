# main.py
import json
from extract_text import extract_text_from_all_files
from topic_modeling import extract_topics
from classify_documents import classify_documents

def process_and_save():
    """Extracts data, classifies documents, and saves results in JSON."""
    extracted_data = extract_text_from_all_files()
    texts = [doc["text"] for doc in extracted_data if doc["text"]]

    topics = extract_topics(texts, num_topics=5)
    classified_data = classify_documents(extracted_data, topics)

    with open("knowledge_base2.json", "w", encoding="utf-8") as f:
        json.dump(classified_data, f, indent=4, ensure_ascii=False)

    print("Data extraction and classification completed successfully!")

if __name__ == "__main__":
    process_and_save()