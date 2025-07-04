import os
from datetime import timezone

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'votre_cle_secrete_ici')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///trash_monitoring.db')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    TRAINING_FOLDER = os.getenv('TRAINING_FOLDER', 'training_data')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    TZ = timezone.utc