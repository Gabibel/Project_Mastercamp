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
    
    # IA
    ai_prediction = db.Column(db.String(50))  # full ou empty
    ai_confidence = db.Column(db.Float)  # 0 ou 1
    ai_validated = db.Column(db.Boolean, default=False) 
    ai_correct = db.Column(db.Boolean)  
    
    # KNN
    knn_prediction = db.Column(db.String(50))  
    knn_confidence = db.Column(db.Float) 
    
    # Random Forest
    rf_prediction = db.Column(db.String(50))
    rf_confidence = db.Column(db.Float)
    
    # SVM
    svm_prediction = db.Column(db.String(50))
    svm_confidence = db.Column(db.Float)
    
    # Vote final ML
    ml_vote = db.Column(db.String(50))
    
    # Métadonnées
    file_size = db.Column(db.Integer)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    avg_color_r = db.Column(db.Float)
    avg_color_g = db.Column(db.Float)
    avg_color_b = db.Column(db.Float)
    contrast = db.Column(db.Float)
    brightness = db.Column(db.Float)
    
    # Nouvelles métadonnées 
    color_variance = db.Column(db.Float)  
    edge_density = db.Column(db.Float)  
    texture_complexity = db.Column(db.Float)  
    dark_pixel_ratio = db.Column(db.Float)  
    color_entropy = db.Column(db.Float)  
    spatial_distribution = db.Column(db.Float) 
    
    latitude = db.Column(db.Float, default=48.8566)
    longitude = db.Column(db.Float, default=2.3522)
    location_name = db.Column(db.String(255), default='Paris')
    
    ml_correct = db.Column(db.Boolean)  

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

KNN_MODEL_FILE = 'knn_model.pkl'
RF_MODEL_FILE  = 'rf_model.pkl'
SVM_MODEL_FILE = 'svm_model.pkl'
