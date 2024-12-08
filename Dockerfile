# Gunakan image base Python 3.12 (sesuai dengan environment yang kamu gunakan)
FROM python:3.12-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Instal semua dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi dari direktori lokal ke dalam container
COPY . .

# Pastikan credential Google Cloud disalin ke dalam container
COPY serviceaccountkey.json /app/serviceaccountkey.json

# Set environment variable untuk credential Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="serviceaccountkey.json"

# Expose port yang digunakan oleh Flask (default 5000)
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]
