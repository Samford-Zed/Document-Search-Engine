import xml.etree.ElementTree as ET
import nltk
import math
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    documents = []
    for study in root.findall('.//clinical_study'):
        brief_title = study.find('brief_title').text
        detailed_description = study.find('detailed_description/textblock').text
        full_text = brief_title + ' ' + detailed_description
        documents.append(full_text)
    
    return documents

# Tokenization, normalization, and stop-word removal
def preprocess_text(text):
    # Lowercasing and removing punctuation
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenizing
    words = nltk.word_tokenize(text)
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Create Inverted Index
def create_inverted_index(documents):
    inverted_index = defaultdict(list)
    doc_freq = defaultdict(int)
    term_freq = defaultdict(int)
    
    for doc_id, doc in enumerate(documents):
        words = doc.split()
        unique_words = set(words)
        for word in unique_words:
            inverted_index[word].append(doc_id)
            doc_freq[word] += 1
        
        for word in words:
            term_freq[(doc_id, word)] += 1
    
    return inverted_index, term_freq, doc_freq

# TF-IDF Calculation
def compute_tf_idf(documents, inverted_index, term_freq, doc_freq):
    num_docs = len(documents)
    tfidf_scores = defaultdict(float)
    
    for (doc_id, word), freq in term_freq.items():
        tf = freq / len(documents[doc_id].split())  # Term frequency
        idf = math.log(num_docs / (1 + doc_freq[word]))  # Inverse Document Frequency
        tfidf_scores[(doc_id, word)] = tf * idf
    
    return tfidf_scores

# Vector Space Model: Cosine Similarity
def compute_cosine_similarity(query, documents, tfidf_scores, inverted_index):
    query_terms = preprocess_text(query).split()
    query_vector = defaultdict(float)
    
    # Vectorize query
    for word in query_terms:
        if word in inverted_index:
            query_vector[word] = query_vector.get(word, 0) + 1  # Simple term frequency for query
    
    # Compute cosine similarity between query and documents
    similarities = {}
    for doc_id, doc in enumerate(documents):
        doc_vector = defaultdict(float)
        for word in query_terms:
            if (doc_id, word) in tfidf_scores:
                doc_vector[word] = tfidf_scores[(doc_id, word)]
        
        # Cosine similarity = (Query . Doc) / (||Query|| * ||Doc||)
        dot_product = sum(query_vector[word] * doc_vector[word] for word in query_terms)
        query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))
        doc_norm = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
        
        if query_norm and doc_norm:
            cosine_similarity = dot_product / (query_norm * doc_norm)
            similarities[doc_id] = cosine_similarity
        else:
            similarities[doc_id] = 0
    
    return similarities

# Main function to execute IR system
def run_ir_system(xml_file, query):
    documents = parse_xml(xml_file)
    
    # Preprocess documents
    documents = [preprocess_text(doc) for doc in documents]
    
    # Build inverted index and calculate TF, DF, and CF
    inverted_index, term_freq, doc_freq = create_inverted_index(documents)
    
    # Compute TF-IDF
    tfidf_scores = compute_tf_idf(documents, inverted_index, term_freq, doc_freq)
    
    # Retrieve documents using query and cosine similarity
    similarities = compute_cosine_similarity(query, documents, tfidf_scores, inverted_index)
    
    # Rank documents based on similarity score
    ranked_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    return ranked_docs

# Example usage
xml_file = "clinical_trial.xml"  # Path to your XML file
query = "aerobic exercise covid"

ranked_docs = run_ir_system(xml_file, query)

print("Ranked Documents:")
for doc_id, score in ranked_docs:
    print(f"Document {doc_id}: Similarity Score: {score}")
