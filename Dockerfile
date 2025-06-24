FROM python:3.12-slim

# Installe les dépendances système nécessaires à OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Crée un répertoire de travail
WORKDIR /app

# Copie les fichiers de ton app
COPY . .

# Installe les packages Python
RUN pip install --upgrade pip && pip install -r requirements.txt

# Lance ton script Python
CMD ["python", "app.py"]
