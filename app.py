# -*- coding: utf-8 -*-
import os
import json
import pickle
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import numpy as np
from collections import defaultdict
from PIL import Image, ImageStat
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import threading
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sqlite3
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'votre_cle_secrete_ici'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trash_monitoring.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRAINING_FOLDER'] = 'training_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

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

def analyze_image_advanced(image_path):
    """Analyse avancée de l'image avec de multiples caractéristiques"""
    try:
        file_size = os.path.getsize(image_path)
        
        # Simulation d'analyse avancée (remplacera PIL/OpenCV plus tard)
        # Pour l'instant, générons des valeurs réalistes basées sur le nom du fichier
        
        filename = os.path.basename(image_path).lower()
        
        # Détection basique basée sur le nom de fichier
        is_full = any(word in filename for word in ['full', 'pleine', 'rempli', 'debord'])
        is_empty = any(word in filename for word in ['empty', 'vide', 'libre', 'disponible'])
        
        if is_full:
            # Caractéristiques d'une poubelle pleine
            brightness = np.random.normal(80, 15)  # Plus sombre
            contrast = np.random.normal(25, 8)     # Moins de contraste
            color_variance = np.random.normal(600, 150)  # Moins de variété
            edge_density = np.random.normal(0.12, 0.03)  # Moins de contours
            texture_complexity = np.random.normal(0.4, 0.1)  # Texture simple
            dark_pixel_ratio = np.random.normal(0.6, 0.1)   # Beaucoup de pixels sombres
            color_entropy = np.random.normal(5.5, 0.5)      # Entropie faible
        elif is_empty:
            # Caractéristiques d'une poubelle vide
            brightness = np.random.normal(160, 20)  # Plus claire
            contrast = np.random.normal(55, 10)     # Plus de contraste
            color_variance = np.random.normal(1400, 200)  # Plus de variété
            edge_density = np.random.normal(0.28, 0.05)   # Plus de contours
            texture_complexity = np.random.normal(0.7, 0.15)  # Texture complexe
            dark_pixel_ratio = np.random.normal(0.15, 0.05)   # Peu de pixels sombres
            color_entropy = np.random.normal(7.2, 0.6)        # Entropie élevée
        else:
            # Valeurs neutres
            brightness = np.random.normal(128, 30)
            contrast = np.random.normal(40, 15)
            color_variance = np.random.normal(1000, 300)
            edge_density = np.random.normal(0.20, 0.08)
            texture_complexity = np.random.normal(0.55, 0.2)
            dark_pixel_ratio = np.random.normal(0.35, 0.15)
            color_entropy = np.random.normal(6.3, 0.8)
        
        # Calcul de la couleur moyenne
        avg_color_r = np.random.normal(128, 30)
        avg_color_g = np.random.normal(128, 30)
        avg_color_b = np.random.normal(128, 30)
        
        # Distribution spatiale (simulation)
        spatial_distribution = np.random.uniform(0.3, 0.8)
        
        return {
            'file_size': file_size,
            'width': 800,
            'height': 600,
            'avg_color_r': float(avg_color_r),
            'avg_color_g': float(avg_color_g),
            'avg_color_b': float(avg_color_b),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'color_variance': float(color_variance),
            'edge_density': float(edge_density),
            'texture_complexity': float(texture_complexity),
            'dark_pixel_ratio': float(dark_pixel_ratio),
            'color_entropy': float(color_entropy),
            'spatial_distribution': float(spatial_distribution)
        }
    except Exception as e:
        print(f"Erreur lors de l'analyse avancée de l'image: {e}")
        return None

RULES_CONFIG_FILE = 'rules_config.json'

def load_rules_config():
    if os.path.exists(RULES_CONFIG_FILE):
        with open(RULES_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def save_rules_config(config):
    with open(RULES_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

@app.route('/')
def index():
    total_images = TrashImage.query.count()
    full_images = TrashImage.query.filter_by(status='full').count()
    empty_images = TrashImage.query.filter_by(status='empty').count()
    pending_images = TrashImage.query.filter_by(status='pending').count()
    
    recent_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    recent_images = TrashImage.query.filter(TrashImage.upload_date >= recent_date).count()
    
    # Statistiques IA supprimées
    return render_template('index.html', 
                         total_images=total_images,
                         full_images=full_images,
                         empty_images=empty_images,
                         pending_images=pending_images,
                         recent_images=recent_images)

def async_analyze_and_update(trash_image_id, filepath):
    from time import sleep
    with app.app_context():
        img = TrashImage.query.get(trash_image_id)
        if not img:
            return
        analysis = analyze_image_advanced(filepath)
        if analysis:
            rule_prediction, rule_confidence = predict_with_advanced_ai(analysis)
            # Prédiction KNN
            knn_pred = None
            knn_conf = None
            rf_pred = None
            rf_conf = None
            svm_pred = None
            svm_conf = None
            feats = [
                analysis['brightness'],
                analysis['contrast'],
                analysis['color_variance'],
                analysis['edge_density'],
                analysis['texture_complexity'],
                analysis['dark_pixel_ratio'],
                analysis['color_entropy']
            ]
            try:
                import pickle
                import numpy as np
                # Charger le scaler
                scaler = None
                if os.path.exists('scaler_ml.pkl'):
                    with open('scaler_ml.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                feats_scaled = scaler.transform([feats]) if scaler else [feats]
                # KNN
                if os.path.exists(KNN_MODEL_FILE):
                    with open(KNN_MODEL_FILE, 'rb') as f:
                        knn = pickle.load(f)
                    proba = knn.predict_proba(feats_scaled)[0]
                    pred = knn.predict(feats_scaled)[0]
                    knn_pred = 'empty' if pred == 0 else 'full'
                    knn_conf = float(np.max(proba))
                # RF
                if os.path.exists(RF_MODEL_FILE):
                    with open(RF_MODEL_FILE, 'rb') as f:
                        rf = pickle.load(f)
                    proba = rf.predict_proba(feats_scaled)[0]
                    pred = rf.predict(feats_scaled)[0]
                    rf_pred = 'empty' if pred == 0 else 'full'
                    rf_conf = float(np.max(proba))
                # SVM
                if os.path.exists(SVM_MODEL_FILE):
                    with open(SVM_MODEL_FILE, 'rb') as f:
                        svm = pickle.load(f)
                    proba = svm.predict_proba(feats_scaled)[0]
                    pred = svm.predict(feats_scaled)[0]
                    svm_pred = 'empty' if pred == 0 else 'full'
                    svm_conf = float(np.max(proba))
            except Exception as e:
                print(f"Erreur prédiction ML : {e}")
            # Vote majoritaire
            votes = [p for p in [knn_pred, rf_pred, svm_pred] if p in ['full', 'empty']]
            if votes:
                ml_vote = max(set(votes), key=votes.count)
            else:
                ml_vote = None
            for k, v in analysis.items():
                setattr(img, k, v)
            img.ai_prediction = rule_prediction
            img.ai_confidence = rule_confidence
            img.knn_prediction = knn_pred
            img.knn_confidence = knn_conf
            img.rf_prediction = rf_pred
            img.rf_confidence = rf_conf
            img.svm_prediction = svm_pred
            img.svm_confidence = svm_conf
            img.ml_vote = ml_vote
            img.status = 'pending'  # Laisse à pending, l'utilisateur valide ensuite
            db.session.commit()

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Aucun fichier sélectionné')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        if not files or all(f.filename == '' for f in files):
            flash('Aucun fichier sélectionné')
            return redirect(request.url)
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        results = []
        for file in files:
            if not file or not file.filename:
                continue
            if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                results.append(f"{file.filename} : format non autorisé")
                continue
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Compression d'image après upload
            try:
                with Image.open(filepath) as img:
                    max_size = 1280
                    if img.width > max_size or img.height > max_size:
                        ratio = min(max_size / img.width, max_size / img.height)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(filepath, format="JPEG", quality=80, optimize=True)
            except Exception as e:
                print(f"Erreur compression image : {e}")
            # Récupérer la localisation du formulaire
            latitude = request.form.get('latitude', type=float)
            longitude = request.form.get('longitude', type=float)
            location_name = request.form.get('location_name', default='Paris')
            # Récupérer la date/heure de la photo
            photo_datetime_str = request.form.get('photo_datetime')
            if photo_datetime_str:
                try:
                    # Format attendu : 'YYYY-MM-DDTHH:MM'
                    photo_datetime = datetime.strptime(photo_datetime_str, '%Y-%m-%dT%H:%M')
                    # On force le fuseau UTC pour cohérence
                    photo_datetime = photo_datetime.replace(tzinfo=timezone.utc)
                except Exception as e:
                    print(f"Erreur parsing date/heure : {e}")
                    photo_datetime = datetime.now(timezone.utc)
            else:
                photo_datetime = datetime.now(timezone.utc)
            # Création entrée TrashImage (status pending, sans features)
            trash_image = TrashImage(
                filename=filename,
                original_filename=file.filename,
                status='pending',
                latitude=latitude if latitude else 48.8566,
                longitude=longitude if longitude else 2.3522,
                location_name=location_name if location_name else 'Paris',
                upload_date=photo_datetime,
            )
            db.session.add(trash_image)
            db.session.commit()
            # Lancer l'analyse en thread
            threading.Thread(target=async_analyze_and_update, args=(trash_image.id, filepath)).start()
            results.append(f"<b>{file.filename}</b> : uploadé, analyse en cours...")
        if results:
            flash('Résultats upload :<br>' + '<br>'.join(results))
        else:
            flash('Aucune image valide uploadée')
        return redirect(url_for('gallery'))
    return render_template('upload.html')

@app.route('/gallery')
def gallery():
    status_filter = request.args.get('status', 'all')
    page = request.args.get('page', 1, type=int)
    
    query = TrashImage.query
    
    if status_filter == 'unknown':
        # Filtrer les images avec prédiction IA 'unknown'
        query = query.filter_by(ai_prediction='unknown')
    elif status_filter != 'all':
        query = query.filter_by(status=status_filter)
    
    images = query.order_by(TrashImage.upload_date.desc()).paginate(
        page=page, per_page=12, error_out=False)
    
    return render_template('gallery.html', images=images, status_filter=status_filter)

@app.route('/validate/<int:image_id>', methods=['POST'])
def validate_prediction(image_id):
    """
    Valider ou corriger la prédiction IA
    """
    image = TrashImage.query.get_or_404(image_id)
    user_status = request.form.get('status')
    
    if user_status in ['full', 'empty']:
        # Mettre à jour le statut
        image.status = user_status
        image.manual_status = user_status
        
        # Marquer comme validé et vérifier si l'IA avait raison
        image.ai_validated = True
        image.ai_correct = (image.ai_prediction == user_status)
        image.ml_correct = (image.ml_vote == user_status)
        db.session.commit()

        if image.ml_correct:
            flash(f'✅ Vote ML correct ! ({image.ml_vote})')
        else:
            flash(f'❌ Vote ML incorrect. Vote: {image.ml_vote}, Réel: {user_status}')
        
        return redirect(url_for('gallery'))
    else:
        flash('Statut invalide')
        return redirect(url_for('gallery'))

@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    image = TrashImage.query.get_or_404(image_id)
    
    # Supprimer le fichier du disque
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Supprimer de la base de données
    db.session.delete(image)
    db.session.commit()
    
    flash("Image supprimée avec succès.")
    return redirect(url_for('gallery'))

@app.route('/dashboard')
def dashboard():
    total_images = TrashImage.query.count()
    # Utiliser le vote final ML pour les stats
    full_images = TrashImage.query.filter_by(ml_vote='full').count()
    empty_images = TrashImage.query.filter_by(ml_vote='empty').count()
    # Statistiques de performance de l'algorithme IA
    validated_images = TrashImage.query.filter_by(ai_validated=True).all()
    ml_correct = sum(1 for img in validated_images if img.ml_correct)
    ml_incorrect = sum(1 for img in validated_images if img.ml_correct is False)
    ml_accuracy = (ml_correct / len(validated_images) * 100) if validated_images else 0
    # Répartition des prédictions IA
    ai_predictions = TrashImage.query.filter(TrashImage.ai_prediction.isnot(None)).all()
    ai_pred_full = sum(1 for img in ai_predictions if img.ai_prediction == 'full')
    ai_pred_empty = sum(1 for img in ai_predictions if img.ai_prediction == 'empty')
    # Incertitude basée sur le vote ML (None ou N/A)
    ai_pred_unknown = sum(1 for img in validated_images if not img.ml_vote or img.ml_vote not in ['full', 'empty'])
    # Statistiques par classe (pleine/vide) selon le vote ML
    full_validated = [img for img in validated_images if img.ml_vote == 'full']
    empty_validated = [img for img in validated_images if img.ml_vote == 'empty']
    full_correct = sum(1 for img in full_validated if img.ml_correct)
    empty_correct = sum(1 for img in empty_validated if img.ml_correct)
    full_accuracy = (full_correct / len(full_validated) * 100) if full_validated else 0
    empty_accuracy = (empty_correct / len(empty_validated) * 100) if empty_validated else 0
    # Comparaison des modèles (KNN, RF, SVM, règles)
    knn_correct = sum(1 for img in validated_images if img.knn_prediction == img.manual_status)
    rf_correct = sum(1 for img in validated_images if img.rf_prediction == img.manual_status)
    svm_correct = sum(1 for img in validated_images if img.svm_prediction == img.manual_status)
    rules_correct = sum(1 for img in validated_images if img.ai_prediction == img.manual_status)
    knn_total = sum(1 for img in validated_images if img.knn_prediction in ['full', 'empty'])
    rf_total = sum(1 for img in validated_images if img.rf_prediction in ['full', 'empty'])
    svm_total = sum(1 for img in validated_images if img.svm_prediction in ['full', 'empty'])
    rules_total = len(validated_images)
    rules_indecis = sum(1 for img in validated_images if img.ai_prediction not in ['full', 'empty'])
    # Compter les images d'entraînement dans les nouveaux dossiers
    empty_training = len([f for f in os.listdir(os.path.join(app.config['TRAINING_FOLDER'], 'with_label', 'clean')) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    full_training = len([f for f in os.listdir(os.path.join(app.config['TRAINING_FOLDER'], 'with_label', 'dirty')) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    return render_template('dashboard.html',
                         total_images=total_images,
                         full_images=full_images,
                         empty_images=empty_images,
                         empty_training=empty_training,
                         full_training=full_training,
                         # Nouvelles statistiques IA
                         ml_correct=ml_correct,
                         ml_incorrect=ml_incorrect,
                         ml_accuracy=ml_accuracy,
                         total_validated=len(validated_images),
                         ai_pred_full=ai_pred_full,
                         ai_pred_empty=ai_pred_empty,
                         ai_pred_unknown=ai_pred_unknown,
                         full_accuracy=full_accuracy,
                         empty_accuracy=empty_accuracy,
                         full_validated_count=len(full_validated),
                         empty_validated_count=len(empty_validated),
                         knn_correct=knn_correct, knn_total=knn_total,
                         rf_correct=rf_correct, rf_total=rf_total,
                         svm_correct=svm_correct, svm_total=svm_total,
                         rules_correct=rules_correct, rules_total=rules_total, rules_indecis=rules_indecis)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    else:
        flash('Fichier non trouvé')
        return redirect(url_for('gallery'))

@app.route('/api/stats')
def api_stats():
    # Récupérer toutes les images
    images = TrashImage.query.all()
    data = []
    for img in images:
        data.append({
            'id': img.id,
            'date': img.upload_date.strftime('%Y-%m-%d'),
            'status': img.status,
            'latitude': img.latitude,
            'longitude': img.longitude,
            'location_name': img.location_name,
            'brightness': img.brightness,
            'contrast': img.contrast,
            'avg_color_r': img.avg_color_r,
            'avg_color_g': img.avg_color_g,
            'avg_color_b': img.avg_color_b,
        })
    return jsonify(data)

@app.route('/static-graph/luminosity')
def static_graph_luminosity():
    images = TrashImage.query.all()
    brightness = [img.brightness for img in images if img.brightness is not None]
    plt.figure(figsize=(6,4))
    plt.hist(brightness, bins=20, color='skyblue')
    plt.title('Histogramme de la luminosité')
    plt.xlabel('Luminosité')
    plt.ylabel('Nombre d\'images')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app.route('/static-graph/status')
def static_graph_status():
    images = TrashImage.query.all()
    status_labels = ['pleine', 'vide', 'en attente']
    status_counts = [
        sum(1 for img in images if img.ml_vote == 'full'),
        sum(1 for img in images if img.ml_vote == 'empty'),
        sum(1 for img in images if img.ml_vote not in ['full', 'empty'])
    ]
    plt.figure(figsize=(5,5))
    plt.pie(status_counts, labels=status_labels, autopct='%1.1f%%', colors=['#e74c3c','#2ecc71','#f1c40f'])
    plt.title('Répartition des statuts (vote ML)')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app.route('/static-graph/contrast')
def static_graph_contrast():
    images = TrashImage.query.all()
    contrast = [img.contrast for img in images if img.contrast is not None]
    plt.figure(figsize=(6,4))
    plt.hist(contrast, bins=20, color='#e67e22')
    plt.title('Histogramme du contraste')
    plt.xlabel('Contraste')
    plt.ylabel('Nombre d\'images')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@app.route('/rules', methods=['GET', 'POST'])
def rules():
    # Valeurs par défaut optimisées
    default_thresholds = {
        'brightness_full_max': 100,      # Plus strict pour pleine
        'brightness_empty_min': 140,     # Plus strict pour vide
        'contrast_full_max': 25,         # Plus strict
        'contrast_empty_min': 40,        # Plus strict
        'color_variance_full_max': 600,  # Plus strict
        'color_variance_empty_min': 1100, # Plus strict
        'edge_density_full_max': 0.12,   # Plus strict
        'edge_density_empty_min': 0.20,  # Plus strict
        'dark_pixel_ratio_full_min': 0.40, # Plus strict
        'dark_pixel_ratio_empty_max': 0.15  # Plus strict
    }
    thresholds = load_rules_config() or default_thresholds
    if request.method == 'POST':
        for k in thresholds:
            val = request.form.get(k)
            if val is not None:
                try:
                    thresholds[k] = float(val)
                except:
                    pass
        save_rules_config(thresholds)
        flash('Règles sauvegardées avec succès.')
        return redirect(url_for('rules'))
    return render_template('rules.html', thresholds=thresholds)

def predict_with_advanced_ai(analysis):
    """
    Prédiction améliorée basée sur des règles plus sophistiquées
    """
    if not analysis:
        return 'unknown', 0.0

    # Poids ajustés pour une meilleure précision
    feature_weights = {
        'brightness': 0.20,      # Augmenté - très important
        'contrast': 0.15,        # Augmenté
        'color_variance': 0.18,  # Stable
        'edge_density': 0.18,    # Réduit légèrement
        'texture_complexity': 0.12,  # Réduit
        'dark_pixel_ratio': 0.12,    # Augmenté
        'color_entropy': 0.05        # Réduit - moins fiable
    }
    
    # Seuils optimisés basés sur l'expérience
    thresholds = load_rules_config() or {
        'brightness_full_max': 100,      # Plus strict pour pleine
        'brightness_empty_min': 140,     # Plus strict pour vide
        'contrast_full_max': 25,         # Plus strict
        'contrast_empty_min': 40,        # Plus strict
        'color_variance_full_max': 600,  # Plus strict
        'color_variance_empty_min': 1100, # Plus strict
        'edge_density_full_max': 0.12,   # Plus strict
        'edge_density_empty_min': 0.20,  # Plus strict
        'dark_pixel_ratio_full_min': 0.40, # Plus strict
        'dark_pixel_ratio_empty_max': 0.15  # Plus strict
    }
    
    features = {
        'brightness': analysis['brightness'],
        'contrast': analysis['contrast'],
        'color_variance': analysis['color_variance'],
        'edge_density': analysis['edge_density'],
        'texture_complexity': analysis['texture_complexity'],
        'dark_pixel_ratio': analysis['dark_pixel_ratio'],
        'color_entropy': analysis['color_entropy']
    }
    
    score_full = 0.0
    score_empty = 0.0
    
    # Règles pour "pleine" - plus strictes
    # Luminosité : poubelle pleine = plus sombre
    if features['brightness'] < thresholds['brightness_full_max']:
        score_full += feature_weights['brightness']
    elif features['brightness'] < thresholds['brightness_full_max'] + 20:  # Zone grise
        score_full += feature_weights['brightness'] * 0.5
    
    # Contraste : poubelle pleine = moins de contraste
    if features['contrast'] < thresholds['contrast_full_max']:
        score_full += feature_weights['contrast']
    elif features['contrast'] < thresholds['contrast_full_max'] + 10:
        score_full += feature_weights['contrast'] * 0.5
    
    # Variance des couleurs : poubelle pleine = moins de variété
    if features['color_variance'] < thresholds['color_variance_full_max']:
        score_full += feature_weights['color_variance']
    elif features['color_variance'] < thresholds['color_variance_full_max'] + 200:
        score_full += feature_weights['color_variance'] * 0.5
    
    # Densité de contours : poubelle pleine = moins de détails
    if features['edge_density'] < thresholds['edge_density_full_max']:
        score_full += feature_weights['edge_density']
    elif features['edge_density'] < thresholds['edge_density_full_max'] + 0.05:
        score_full += feature_weights['edge_density'] * 0.5
    
    # Ratio de pixels sombres : poubelle pleine = plus de zones sombres
    if features['dark_pixel_ratio'] > thresholds['dark_pixel_ratio_full_min']:
        score_full += feature_weights['dark_pixel_ratio']
    elif features['dark_pixel_ratio'] > thresholds['dark_pixel_ratio_full_min'] - 0.1:
        score_full += feature_weights['dark_pixel_ratio'] * 0.5
    
    # Complexité de texture : poubelle pleine = texture plus simple
    if features['texture_complexity'] < 0.5:
        score_full += feature_weights['texture_complexity']
    
    # Règles pour "vide" - plus strictes
    # Luminosité : poubelle vide = plus claire
    if features['brightness'] > thresholds['brightness_empty_min']:
        score_empty += feature_weights['brightness']
    elif features['brightness'] > thresholds['brightness_empty_min'] - 20:  # Zone grise
        score_empty += feature_weights['brightness'] * 0.5
    
    # Contraste : poubelle vide = plus de contraste
    if features['contrast'] > thresholds['contrast_empty_min']:
        score_empty += feature_weights['contrast']
    elif features['contrast'] > thresholds['contrast_empty_min'] - 10:
        score_empty += feature_weights['contrast'] * 0.5
    
    # Variance des couleurs : poubelle vide = plus de variété
    if features['color_variance'] > thresholds['color_variance_empty_min']:
        score_empty += feature_weights['color_variance']
    elif features['color_variance'] > thresholds['color_variance_empty_min'] - 200:
        score_empty += feature_weights['color_variance'] * 0.5
    
    # Densité de contours : poubelle vide = plus de détails
    if features['edge_density'] > thresholds['edge_density_empty_min']:
        score_empty += feature_weights['edge_density']
    elif features['edge_density'] > thresholds['edge_density_empty_min'] - 0.05:
        score_empty += feature_weights['edge_density'] * 0.5
    
    # Ratio de pixels sombres : poubelle vide = moins de zones sombres
    if features['dark_pixel_ratio'] < thresholds['dark_pixel_ratio_empty_max']:
        score_empty += feature_weights['dark_pixel_ratio']
    elif features['dark_pixel_ratio'] < thresholds['dark_pixel_ratio_empty_max'] + 0.1:
        score_empty += feature_weights['dark_pixel_ratio'] * 0.5
    
    # Complexité de texture : poubelle vide = texture plus complexe
    if features['texture_complexity'] > 0.6:
        score_empty += feature_weights['texture_complexity']
    
    # Règles combinées pour améliorer la précision
    # Si plusieurs indicateurs pointent dans la même direction, bonus
    indicators_full = 0
    indicators_empty = 0
    
    if features['brightness'] < thresholds['brightness_full_max']:
        indicators_full += 1
    if features['contrast'] < thresholds['contrast_full_max']:
        indicators_full += 1
    if features['dark_pixel_ratio'] > thresholds['dark_pixel_ratio_full_min']:
        indicators_full += 1
    
    if features['brightness'] > thresholds['brightness_empty_min']:
        indicators_empty += 1
    if features['contrast'] > thresholds['contrast_empty_min']:
        indicators_empty += 1
    if features['dark_pixel_ratio'] < thresholds['dark_pixel_ratio_empty_max']:
        indicators_empty += 1
    
    # Bonus de cohérence
    if indicators_full >= 2:
        score_full += 0.1
    if indicators_empty >= 2:
        score_empty += 0.1
    
    # Normalisation et calcul de confiance
    total_weight = sum(feature_weights.values()) + 0.2  # +0.2 pour les bonus
    score_full = min(0.95, score_full / total_weight)
    score_empty = min(0.95, score_empty / total_weight)
    
    # Décision avec seuil de confiance minimum
    if score_full > score_empty and score_full > 0.3:  # Seuil minimum de confiance
        confidence = score_full + (score_full - score_empty) * 0.3
        return 'full', min(0.95, confidence)
    elif score_empty > score_full and score_empty > 0.3:
        confidence = score_empty + (score_empty - score_full) * 0.3
        return 'empty', min(0.95, confidence)
    else:
        # Si pas assez de confiance, retourner "unknown"
        return 'unknown', max(score_full, score_empty)

@app.route('/audit')
def audit():
    images = TrashImage.query.order_by(TrashImage.upload_date.desc()).all()
    audit_results = []
    for img in images:
        anomalies = []
        # Vérifier la présence du fichier
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        if not os.path.exists(filepath):
            anomalies.append("Fichier manquant")
        # Vérifier dimensions et taille
        if not img.width or not img.height or img.width <= 0 or img.height <= 0:
            anomalies.append("Dimensions invalides")
        if not img.file_size or img.file_size <= 0:
            anomalies.append("Taille fichier nulle")
        # Vérifier caractéristiques principales
        for field in ["brightness", "contrast", "color_variance", "edge_density", "dark_pixel_ratio"]:
            val = getattr(img, field, None)
            if val is None:
                anomalies.append(f"{field} manquant")
        # Statut incohérent
        if img.status not in ["full", "empty", "pending"]:
            anomalies.append("Statut inconnu")
        if not img.ai_prediction:
            anomalies.append("Pas de prédiction")
        audit_results.append({
            "image": img,
            "anomalies": anomalies
        })
    # Audit rapide des features : moyenne et écart-type par classe
    features = ["brightness", "contrast", "color_variance", "edge_density", "texture_complexity", "dark_pixel_ratio", "color_entropy"]
    stats = {}
    for classe in ["full", "empty"]:
        classe_imgs = [img for img in images if img.status == classe]
        stats[classe] = {}
        for feat in features:
            vals = [getattr(img, feat, None) for img in classe_imgs if getattr(img, feat, None) is not None]
            vals = [v for v in vals if v is not None]  # Filtre les None pour éviter l'erreur numpy
            if vals:
                stats[classe][feat] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "count": len(vals)
                }
            else:
                stats[classe][feat] = {"mean": None, "std": None, "count": 0}
    return render_template('audit.html', audit_results=audit_results, feature_stats=stats)

@app.route('/api/map_data')
def api_map_data():
    images = TrashImage.query.all()
    data = []
    for img in images:
        data.append({
            'id': img.id,
            'latitude': img.latitude,
            'longitude': img.longitude,
            'status': img.status,
            'ai_prediction': img.ai_prediction,
            'upload_date': img.upload_date.strftime('%Y-%m-%d %H:%M'),
            'location_name': img.location_name,
            'filename': img.filename,
            'url': url_for('uploaded_file', filename=img.filename)
        })
    return jsonify(data)

KNN_MODEL_FILE = 'knn_model.pkl'
RF_MODEL_FILE = 'rf_model.pkl'
SVM_MODEL_FILE = 'svm_model.pkl'

# Fonction d'extraction des features pour une image (utilise analyze_image_advanced)
def extract_features_for_knn(image_path):
    features = analyze_image_advanced(image_path)
    if not features:
        return None
    return [
        features['brightness'],
        features['contrast'],
        features['color_variance'],
        features['edge_density'],
        features['texture_complexity'],
        features['dark_pixel_ratio'],
        features['color_entropy']
    ]

# Entraînement du modèle KNN, RandomForest et SVM à partir des images d'entraînement
def train_ml_models():
    X = []
    y = []
    # Images vides (clean)
    clean_dir = os.path.join(app.config['TRAINING_FOLDER'], 'with_label', 'clean')
    for img_file in glob.glob(os.path.join(clean_dir, '*')):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            feats = extract_features_for_knn(img_file)
            if feats:
                X.append(feats)
                y.append(0)  # 0 = empty
    # Images pleines (dirty)
    dirty_dir = os.path.join(app.config['TRAINING_FOLDER'], 'with_label', 'dirty')
    for img_file in glob.glob(os.path.join(dirty_dir, '*')):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            feats = extract_features_for_knn(img_file)
            if feats:
                X.append(feats)
                y.append(1)  # 1 = full
    if not X:
        print('Aucune donnée pour entraîner les modèles ML.')
        return False
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)
    with open(KNN_MODEL_FILE, 'wb') as f:
        pickle.dump(knn, f)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_scaled, y)
    with open(RF_MODEL_FILE, 'wb') as f:
        pickle.dump(rf, f)
    # SVM (avec probas)
    svm = SVC(probability=True, kernel='rbf', random_state=42)
    svm.fit(X_scaled, y)
    with open(SVM_MODEL_FILE, 'wb') as f:
        pickle.dump(svm, f)
    # Sauvegarde du scaler
    with open('scaler_ml.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f'Modèles ML entraînés et sauvegardés ({len(X)} images).')
    return True

# Route admin pour entraîner les modèles ML manuellement
@app.route('/train_knn')
def train_knn_route():
    ok = train_ml_models()
    if ok:
        flash('Modèles ML (KNN, RF, SVM) entraînés avec succès.')
    else:
        flash('Erreur lors de l\'entraînement des modèles ML (pas assez de données).')
    return redirect(url_for('rules'))

@app.route('/reset_rules_knn')
def reset_rules_knn():
    import numpy as np
    # Collecte des features par classe
    clean_dir = os.path.join(app.config['TRAINING_FOLDER'], 'with_label', 'clean')
    dirty_dir = os.path.join(app.config['TRAINING_FOLDER'], 'with_label', 'dirty')
    clean_feats = []
    dirty_feats = []
    for img_file in glob.glob(os.path.join(clean_dir, '*')):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            feats = extract_features_for_knn(img_file)
            if feats:
                clean_feats.append(feats)
    for img_file in glob.glob(os.path.join(dirty_dir, '*')):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            feats = extract_features_for_knn(img_file)
            if feats:
                dirty_feats.append(feats)
    if not clean_feats or not dirty_feats:
        flash("Pas assez de données pour réinitialiser les règles.")
        return redirect(url_for('rules'))
    clean_feats = np.array(clean_feats)
    dirty_feats = np.array(dirty_feats)
    # Calcul des seuils optimaux (moyenne entre les deux classes)
    thresholds = {
        'brightness_full_max': float(np.max(dirty_feats[:,0])),
        'brightness_empty_min': float(np.min(clean_feats[:,0])),
        'contrast_full_max': float(np.max(dirty_feats[:,1])),
        'contrast_empty_min': float(np.min(clean_feats[:,1])),
        'color_variance_full_max': float(np.max(dirty_feats[:,2])),
        'color_variance_empty_min': float(np.min(clean_feats[:,2])),
        'edge_density_full_max': float(np.max(dirty_feats[:,3])),
        'edge_density_empty_min': float(np.min(clean_feats[:,3])),
        'dark_pixel_ratio_full_min': float(np.min(dirty_feats[:,5])),
        'dark_pixel_ratio_empty_max': float(np.max(clean_feats[:,5]))
    }
    save_rules_config(thresholds)
    flash("Règles réinitialisées selon les statistiques du KNN.")
    return redirect(url_for('rules'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 