import uuid
from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import logging

app = Flask(__name__)
BUCKET_NAME = 'storage-ml-similliar'
CSV_FILE_PATH = 'dataset-book/merged_books_ratings (3).csv'
MODEL_PATH = 'model-book/book_recommendation_model.h5'
LOCAL_MODEL_PATH = '/tmp/book_recommendation_model.h5'

logging.basicConfig(level=logging.INFO)

# Fungsi untuk inisialisasi Firestore
def initialize_firestore():
    try:
        db = firestore.Client()
        logging.info("Firestore initialized successfully.")
        return db
    except Exception as e:
        logging.error(f"Error initializing Firestore: {e}")
        return None

db = initialize_firestore()
if db is None:
    logging.error("Firestore initialization failed. Exiting.")
    exit(1)

# Fungsi untuk mendapatkan buku dari Firestore
def get_books_from_firestore():
    try:
        books_ref = db.collection('books')
        books = books_ref.stream()
        books_data = [book.to_dict() for book in books]
        return books_data
    except Exception as e:
        logging.error(f"Error fetching data from Firestore: {e}")
        return []

# Fungsi untuk mengunduh file dari GCS
def download_blob(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_text()

# Fungsi untuk mengunggah CSV ke GCS
def upload_csv(bucket_name, destination_blob_name, data_frame):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

# Fungsi untuk mengunggah file ke GCS
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")

# Fungsi untuk mengunduh model dari GCS
def download_model(bucket_name, model_path, local_model_path):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        blob.download_to_filename(local_model_path)
        logging.info(f"Model downloaded successfully to {local_model_path}.")
    except Exception as e:
        logging.error(f"Error downloading model from GCS: {e}")
        raise e

try:
    # Mengunduh model ke direktori sementara
    download_model(BUCKET_NAME, MODEL_PATH, LOCAL_MODEL_PATH)
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    logging.info("ML model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading ML model: {e}")
    exit(1)

try:
    # Memuat data CSV dari GCS
    csv_data = download_blob(BUCKET_NAME, CSV_FILE_PATH)
    df = pd.read_csv(StringIO(csv_data))
    logging.info("CSV data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading CSV data: {e}")
    df = pd.DataFrame()

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

    # Simpan file sementara
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)
    
    # Unggah file ke Google Cloud Storage
    upload_blob(BUCKET_NAME, temp_path, file.filename)

    valid_genres = [
        'fiction', 'non-fiction', 'children\'s', 'education', 'religion', 
        'comics', 'art', 'health', 'technology', 'other'
    ]

    try:
        # Validasi genre
        general_category = request.form['general_category']
        if general_category not in valid_genres:
            return jsonify({'error': f'Invalid genre. Valid genres are: {", ".join(valid_genres)}'}), 400

        # Informasi buku
        book_info = {
            'Title': request.form['Title'],
            'id': str(uuid.uuid4()),
            'authors': request.form['author'],
            'review/score': float(request.form['rating']),
            'profil_Name': request.form['profil_Name'],
            'general_category': general_category
        }
        logging.info(f"Book Info: {book_info}")

        # Simpan ke Firestore
        db.collection('books').document(book_info['id']).set(book_info)

        # Update CSV
        global df
        new_row = {
            'profil_Name': book_info['profil_Name'],
            'Title': book_info['Title'],
            'review/score': book_info['review/score'],
            'general_category': book_info['general_category']
        }

        if 'Genre' in df.columns:
            df.rename(columns={'Genre': 'general_category'}, inplace=True)

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        upload_csv(BUCKET_NAME, CSV_FILE_PATH, df)

    except Exception as e:
        logging.error(f"Error in /upload: {e}")
        return jsonify({'error': f'Failed to process data: {e}'}), 500
    finally:
        os.remove(temp_path)

    return jsonify({'message': 'File uploaded successfully', 'book_info': book_info}), 200

# Endpoint untuk mendapatkan data buku
@app.route('/get_buku', methods=['GET'])
def get_buku():
    columns_to_remove = ['Genre', 'title', 'user']
    df_filtered = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
    books_data_firestore = get_books_from_firestore()
    books_data_gcs = df_filtered.to_dict(orient='records')
    all_books_data = books_data_gcs + books_data_firestore
    return jsonify({"message": "Daftar Buku", "data": all_books_data})

# Fungsi untuk mendapatkan rekomendasi buku menggunakan model TensorFlow
def rekomendasi_ml(book_title, model, df, top_n=5):
    # Pastikan kolom 'Title' tidak memiliki nilai NaN
    df = df.dropna(subset=['Title'])

    # Ambil daftar judul buku dari kolom 'Title'
    book_titles = df['Title'].tolist()

    # Pastikan judul buku yang diminta ada di dataset
    if book_title not in book_titles:
        return jsonify({'error': f"Buku berjudul '{book_title}' tidak ditemukan dalam database."}), 404

    # Dapatkan embedding atau representasi untuk buku yang diminta
    try:
        book_index = book_titles.index(book_title)
        book_data = np.array([book_index])  # Ini hanya contoh, sesuaikan dengan model kamu
        book_embedding = model.predict(book_data)  # Dapatkan embedding dari model
    except Exception as e:
        logging.error(f"Error generating recommendation: {e}")
        return jsonify({'error': 'Failed to generate recommendation'}), 500

    # Bandingkan embedding buku tersebut dengan semua embedding lainnya untuk mendapatkan rekomendasi
    all_embeddings = model.predict(np.arange(len(book_titles)))  # Mendapatkan embedding untuk semua buku

    # Hitung kesamaan kosinus antara embedding buku yang diminta dengan buku lainnya
    cosine_sim = cosine_similarity(book_embedding, all_embeddings).flatten()

    # Urutkan buku berdasarkan skor kesamaan (kecuali buku itu sendiri)
    sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)

    # Ambil indeks buku yang paling mirip (kecuali buku pertama yang merupakan buku itu sendiri)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Kembalikan daftar judul buku yang paling mirip
    rekomendasi_buku = df['Title'].iloc[sim_indices].tolist()

    return jsonify({'rekomendasi': rekomendasi_buku})

# Contoh penggunaan di route Flask
@app.route('/rekomendasi', methods=['GET'])
def get_rekomendasi():
    book_title = request.args.get('book_title', default="", type=str)
    if not book_title:
        return jsonify({'error': 'Judul buku harus diberikan.'}), 400

    # Panggil fungsi rekomendasi dengan model TensorFlow
    return rekomendasi_ml(book_title, model, df)

# Endpoint rating buku
@app.route('/rating', methods=['POST'])
def rating():
    try:
        data = request.json
        book_name = data.get('Title', None)
        if not book_name:
            return jsonify({'error': 'Missing key: title'}), 400

        if 'Title' not in df.columns or 'review/score' not in df.columns:
            return jsonify({'error': 'Required columns not found in dataset'}), 500

        review_scores = df[df['Title'].str.lower() == book_name.lower()]['review/score']
        if review_scores.empty:
            return jsonify({'error': 'No reviews found for this book'}), 404

        average_rating = review_scores.mean()
        min_rating = df['review/score'].min()
        max_rating = df['review/score'].max()
        normalized_score = (average_rating - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0

        return jsonify({'average_rating': average_rating, 'normalized_score': normalized_score}), 200
    except Exception as e:
        logging.error(f"Error in /rating: {e}")
        return jsonify({'error': f'Failed to calculate rating: {e}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
