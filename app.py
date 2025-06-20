from flask import Flask, render_template, request
import os
import xml.etree.ElementTree as ET
import math
import string
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

app = Flask(__name__, static_folder='assets', static_url_path='/assets')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [stemmer.stem(token) for token in tokens if token not in stop_words]

def load_documents(folder_path='documents'):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                content = " ".join([elem.text for elem in root.iter() if elem.text])
                tokens = preprocess(content)
                docs[filename] = {
                    'tokens': tokens,
                    'content': content,
                    'filename': filename  
                }
            except ET.ParseError:
                continue
    return docs

def build_index(docs):
    index = defaultdict(lambda: defaultdict(int))
    df = defaultdict(int)
    cf = defaultdict(int)
    for doc_id, data in docs.items():
        counted = Counter(data['tokens'])
        for term, freq in counted.items():
            index[term][doc_id] = freq
            df[term] += 1
            cf[term] += freq
    return index, df, cf

def compute_tfidf_vector(tokens, df, N):
    tf = Counter(tokens)
    vector = {}
    for term, freq in tf.items():
        if df.get(term):
            idf = math.log(N / df[term])
            vector[term] = freq * idf
    return vector, tf

def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in vec1)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    docs = load_documents()
    index_data, df, cf = build_index(docs)
    total_docs = len(docs)
    query = request.form.get('query', "") if request.method == 'POST' else request.args.get('query', "")
    page = int(request.args.get('page', 1))
    per_page = 5
    results = []
    total_result_count = 0

    if query:
        q_tokens = preprocess(query)
        query_vector, query_tf = compute_tfidf_vector(q_tokens, df, total_docs)

        for doc_id, data in docs.items():
            doc_vector, doc_tf = compute_tfidf_vector(data['tokens'], df, total_docs)
            sim = cosine_similarity(query_vector, doc_vector)
            
            tfidf_details = []
            for term in q_tokens:
                tf = doc_tf.get(term, 0)
                idf = math.log(total_docs / (df[term] + 1)) if df.get(term) else 0
                weight = tf * idf
                tfidf_details.append({
                    'term': term,
                    'tf': tf,
                    'idf': round(idf, 4),
                    'weight': round(weight, 4)
                })
            
            results.append({
                'filename': data['filename'], 
                'similarity': max(0, round(sim, 4)), 
                'length': len(data['tokens']),
                'content': data['content'],
                'tfidf_details': tfidf_details
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        for rank, result in enumerate(results, start=1):
            result['rank'] = rank
        total_result_count = len(results)

    start = (page - 1) * per_page
    end = start + per_page
    if not results:
        paginated_results = []
        for data in list(docs.values())[start:end]:
            paginated_results.append({
                'filename': data['filename'],
                'similarity': 0.0,
                'length': len(data['tokens']),
                'content': data['content'],
                'tfidf_details': []
            })
    else:
        paginated_results = results[start:end]

    total_pages = math.ceil((len(results) if results else len(docs)) / per_page)

    return render_template(
        'index.html',
        results=paginated_results,
        query=query,
        page=page,
        total_pages=total_pages,
        show_fallback=not bool(results) and bool(query),
        total_result_count=total_result_count
    )

if __name__ == '__main__':
    app.run(debug=True)