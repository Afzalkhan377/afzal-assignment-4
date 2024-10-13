from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load the dataset and stopwords
newsgroups = fetch_20newsgroups(subset='all')
stop_words = stopwords.words('english')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# Apply LSA (Truncated SVD)
n_components = 100  # Number of components for dimensionality reduction
svd_model = TruncatedSVD(n_components=n_components, random_state=42)
X_lsa = svd_model.fit_transform(X_tfidf)

# Store the documents in LSA space
document_matrix = X_lsa

# Search engine function
def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform the query into the same TF-IDF space
    query_tfidf = vectorizer.transform([query])
    
    # Project the query into the LSA space
    query_lsa = svd_model.transform(query_tfidf)
    
    # Compute cosine similarity between the query and the document matrix
    similarities = cosine_similarity(query_lsa, document_matrix)[0]
    
    # Get top 5 most similar documents
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_similarities = similarities[top_indices]
    top_documents = [newsgroups.data[i] for i in top_indices]
    
    return top_documents, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    
    # Convert numpy arrays to lists
    similarities = similarities.tolist()
    indices = indices.tolist()
    
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})

if __name__ == '__main__':
    app.run(debug=True)
