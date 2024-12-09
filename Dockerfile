# Gunakan image dasar Python
FROM python:3.9-slim

# Tentukan direktori kerja dalam container
WORKDIR /app

# Salin file aplikasi ke dalam container
COPY . /app

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Tentukan port yang digunakan oleh aplikasi
ENV PORT=8080
EXPOSE 8080

# Perintah untuk menjalankan aplikasi
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
