from app import db
import datetime
from datetime import timezone
class TrashImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    status = db.Column(db.String(50), default='pending')
    manual_status = db.Column(db.String(50))
    
    # Prédiction IA
    ai_prediction = db.Column(db.String(50))  # 'full' ou 'empty'
    ai_confidence = db.Column(db.Float)  # Score de confiance (0-1)
    ai_validated = db.Column(db.Boolean, default=False)  # Si l'IA a été validée
    ai_correct = db.Column(db.Boolean)  # Si la prédiction IA était correcte
    
    # Prédiction KNN
    knn_prediction = db.Column(db.String(50))  # 'full' ou 'empty'
    knn_confidence = db.Column(db.Float)  # Score de confiance (0-1)
    
    # Prédiction Random Forest
    rf_prediction = db.Column(db.String(50))
    rf_confidence = db.Column(db.Float)
    
    # Prédiction SVM
    svm_prediction = db.Column(db.String(50))
    svm_confidence = db.Column(db.Float)
    
    # Vote final ML
    ml_vote = db.Column(db.String(50))
    
    # Métadonnées d'analyse avancée
    file_size = db.Column(db.Integer)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    avg_color_r = db.Column(db.Float)
    avg_color_g = db.Column(db.Float)
    avg_color_b = db.Column(db.Float)
    contrast = db.Column(db.Float)
    brightness = db.Column(db.Float)
    
    # Nouvelles métadonnées pour IA améliorée
    color_variance = db.Column(db.Float)  # Variance des couleurs
    edge_density = db.Column(db.Float)  # Densité de contours
    texture_complexity = db.Column(db.Float)  # Complexité de texture
    dark_pixel_ratio = db.Column(db.Float)  # Ratio de pixels sombres
    color_entropy = db.Column(db.Float)  # Entropie des couleurs
    spatial_distribution = db.Column(db.Float)  # Distribution spatiale
    
    latitude = db.Column(db.Float, default=48.8566)
    longitude = db.Column(db.Float, default=2.3522)
    location_name = db.Column(db.String(255), default='Paris')
    
    # Ajout du champ ml_correct dans TrashImage
    ml_correct = db.Column(db.Boolean)  # Si le vote ML était correct

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

KNN_MODEL_FILE = 'knn_model.pkl'
RF_MODEL_FILE  = 'rf_model.pkl'
SVM_MODEL_FILE = 'svm_model.pkl'
