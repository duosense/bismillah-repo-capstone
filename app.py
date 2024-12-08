from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

app = Flask(__name__)  # Perbaikan di sini, mengganti _name_ menjadi __name__

# Konfigurasi
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceaccountkey.json'
BUCKET_NAME = 'storage-ml-similliar'
CSV_FILE_PATH = 'dataset-book/merged_books_ratings (2).csv'

# Inisialisasi Firestore
def initialize_firestore():
    try:
        db = firestore.Client()
        print("Firestore initialized successfully.")
        return db
    except Exception as e:
        print(f"Error initializing Firestore: {e}")
        return None

db = initialize_firestore()
if db is None:
    print("Firestore initialization failed. Exiting.")
    exit(1)

# Inisialisasi model ML
try:
    model = tf.keras.models.load_model('book_recommendation_model.h5')
except Exception as e:
    print(f"Error loading ML model: {e}")
    exit(1)

# Fungsi utilitas
def download_blob(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_text()

def upload_csv(bucket_name, destination_blob_name, data_frame):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Load CSV data
csv_data = download_blob(BUCKET_NAME, CSV_FILE_PATH)
df = pd.read_csv(StringIO(csv_data))

@app.route("/")
def index():
    return jsonify({"status": {"code": 200, "message": "API is running"}}), 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.form:
        return jsonify({'error': 'Invalid request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)
    upload_blob(BUCKET_NAME, temp_path, file.filename)

    try:
        # Validasi dan log
        book_info = {
            'name': request.form['name'],
            'id': request.form['id'],
            'author': request.form['author'],
            'rating': float(request.form['rating']),
            'user': request.form['user']
        }
        print("Book Info:", book_info)

        # Simpan ke Firestore
        db.collection('books').document(book_info['id']).set(book_info)

        # Update DataFrame
        global df
        new_row = {
            'user': book_info['user'],
            'book': book_info['id'],
            'review/score': book_info['rating']
        }
        print("New Row:", new_row)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Upload CSV
        upload_csv(BUCKET_NAME, CSV_FILE_PATH, df)

    except Exception as e:
        print("Error in /upload:", e)
        return jsonify({'error': f'Failed to process data: {e}'}), 500
    finally:
        os.remove(temp_path)

    return jsonify({'message': 'File uploaded successfully', 'book_info': book_info}), 200


@app.route('/get_buku', methods=['GET'])
def get_buku():
    try:
        books = db.collection('books').stream()
        book_list = [book.to_dict() for book in books]
        print("Books fetched:", book_list)  # Debug log
        return jsonify({'message': 'Daftar buku', 'data': book_list}), 200
    except Exception as e:
        print("Error in /get_buku:", e)  # Debug log
        return jsonify({'error': f'Failed to fetch books: {e}'}), 500

# Fungsi untuk mendapatkan rekomendasi buku
def get_top_recommendations(book_title, df, cosine_sim, top_n=5):
    if book_title not in df['Title'].values:
        return {"error": "Buku tidak ditemukan!"}

    book_idx = df[df['Title'] == book_title].index[0]
    similarity_scores = cosine_sim[book_idx]
    similar_books_idx = similarity_scores.argsort()[-top_n-1:-1][::-1]

    recommendations = []
    for idx in similar_books_idx:
        recommendations.append({
            'Title': df.iloc[idx]['Title'],
            'Similarity Score': similarity_scores[idx]
        })

    return recommendations

# Fungsi untuk menghitung cosine similarity dari embeddings buku
def calculate_cosine_similarity(book_embeddings):
    # Meratakan seluruh embeddings menjadi 2D
    book_embeddings_flattened = book_embeddings.reshape(book_embeddings.shape[0], -1)
    cosine_sim = cosine_similarity(book_embeddings_flattened)
    return cosine_sim

@app.route('/recommendations', methods=['GET'])
def recommendations():
    # Mendapatkan judul buku dari query parameter
    book_title = request.args.get('book_title')
    
    # Menghitung embeddings buku berdasarkan deskripsi buku
    book_embeddings = model.predict(df['description'].values)  # Model menghasilkan embeddings dari deskripsi buku
    
    # Menghitung cosine similarity berdasarkan embeddings buku
    cosine_sim = calculate_cosine_similarity(book_embeddings)
    
    # Mendapatkan rekomendasi buku
    recommendations_result = get_top_recommendations(book_title, df, cosine_sim, top_n=5)
    
    return jsonify(recommendations_result)

@app.route('/rating', methods=['POST'])
def rating():
    try:
        data = request.json
        user = data['user']
        book = data['book']

        review_score = df[(df['user'] == user) & (df['book'] == book)]['review/score']
        if review_score.empty:
            return jsonify({'error': 'No review found'}), 404

        min_rating = df['review/score'].min()
        max_rating = df['review/score'].max()
        normalized_score = (review_score.values[0] - min_rating) / (max_rating - min_rating)
        return jsonify({'normalized_score': normalized_score}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to calculate rating: {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True)