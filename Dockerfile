FROM python:3.12-slim

# Installer les bibliothèques nécessaires à OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Créer le répertoire de travail
WORKDIR /app

# Copier le code source dans le conteneur
COPY . .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Lancer l'application
CMD ["python", "app.py"]
