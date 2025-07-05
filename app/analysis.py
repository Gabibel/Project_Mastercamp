from PIL import Image, ImageStat
import numpy as np
import os 
import cv2
import json
import pickle
def analyze_image_advanced(image_path):
    """Analyse avancée de l'image avec de multiples caractéristiques réelles"""
    try:
        file_size = os.path.getsize(image_path)
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        stat = ImageStat.Stat(img)
        avg_color_r, avg_color_g, avg_color_b = stat.mean
        # Calcul de la luminance (perçue)
        np_img = np.array(img)
        luminance = 0.2126 * np_img[:,:,0] + 0.7152 * np_img[:,:,1] + 0.0722 * np_img[:,:,2]
        brightness = float(np.mean(luminance))
        contrast = float(np.std(luminance))
        # Variance des couleurs (sur tous les canaux)
        color_variance = float(np.var(np_img))
        # Densité de contours (OpenCV Canny)
        img_gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        edge_density = float(np.sum(edges > 0)) / (width * height)
        # Complexité de texture (std du Laplacien)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        texture_complexity = float(np.std(laplacian)) / 100.0  # normalisé
        # Ratio de pixels sombres (luminance < 50)
        dark_pixel_ratio = float(np.sum(luminance < 50)) / (width * height)
        # Entropie des couleurs (histogramme sur 256 bins)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0,256])
        hist_norm = hist / hist.sum()
        color_entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-7)))
        # Distribution spatiale (std des positions des pixels sombres)
        dark_pixels = np.argwhere(luminance < 50)
        if len(dark_pixels) > 0:
            spatial_distribution = float(np.std(dark_pixels[:,0]) + np.std(dark_pixels[:,1])) / (width + height)
        else:
            spatial_distribution = 0.0
        return {
            'file_size': file_size,
            'width': width,
            'height': height,
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

def predict_with_advanced_ai(analysis):
    """
    Prédiction améliorée basée sur des règles plus sophistiquées et équilibrées, avec détection des poubelles fermées.
    """
    if not analysis:
        return 'unknown', 0.0

    # Détection poubelle fermée : très peu de texture et de contours
    if analysis['texture_complexity'] < 0.18 and analysis['edge_density'] < 0.07:
        return 'unknown', 0.0

    feature_weights = {
        'brightness': 0.04,
        'contrast': 0.13,
        'color_variance': 0.13,
        'edge_density': 0.32,
        'texture_complexity': 0.20,
        'dark_pixel_ratio': 0.06,
        'color_entropy': 0.13
    }

    thresholds = load_rules_config() or {
        "brightness_full_max": 171.02509750353522,
        "brightness_empty_min": 64.52661851117642,
        "contrast_full_max": 70.39170486727024,
        "contrast_empty_min": 18.930505417541436,
        "color_variance_full_max": 1541.416145367293,
        "color_variance_empty_min": 373.91045763036334,
        "edge_density_full_max": 0.33342089399350083,
        "edge_density_empty_min": -0.051013907901757416,
        "dark_pixel_ratio_full_min": 0.1570235406600966,
        "dark_pixel_ratio_empty_max": 0.6985804852032718
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

    # Règles pour "pleine"
    if features['brightness'] < thresholds['brightness_full_max']:
        score_full += feature_weights['brightness']
    elif features['brightness'] < thresholds['brightness_full_max'] + 20:
        score_full += feature_weights['brightness'] * 0.5

    if features['contrast'] < thresholds['contrast_full_max']:
        score_full += feature_weights['contrast']
    elif features['contrast'] < thresholds['contrast_full_max'] + 10:
        score_full += feature_weights['contrast'] * 0.5

    if features['color_variance'] < thresholds['color_variance_full_max']:
        score_full += feature_weights['color_variance']
    elif features['color_variance'] < thresholds['color_variance_full_max'] + 200:
        score_full += feature_weights['color_variance'] * 0.5

    if features['edge_density'] < thresholds['edge_density_full_max']:
        score_full += feature_weights['edge_density']
    elif features['edge_density'] < thresholds['edge_density_full_max'] + 0.04:
        score_full += feature_weights['edge_density'] * 0.5

    if features['dark_pixel_ratio'] > thresholds['dark_pixel_ratio_full_min']:
        score_full += feature_weights['dark_pixel_ratio']
    elif features['dark_pixel_ratio'] > thresholds['dark_pixel_ratio_full_min'] - 0.08:
        score_full += feature_weights['dark_pixel_ratio'] * 0.5

    if features['texture_complexity'] < 0.55:
        score_full += feature_weights['texture_complexity']

    # Règles pour "vide"
    if features['brightness'] > thresholds['brightness_empty_min']:
        score_empty += feature_weights['brightness']
    elif features['brightness'] > thresholds['brightness_empty_min'] - 20:
        score_empty += feature_weights['brightness'] * 0.5

    if features['contrast'] > thresholds['contrast_empty_min']:
        score_empty += feature_weights['contrast']
    elif features['contrast'] > thresholds['contrast_empty_min'] - 10:
        score_empty += feature_weights['contrast'] * 0.5

    if features['color_variance'] > thresholds['color_variance_empty_min']:
        score_empty += feature_weights['color_variance']
    elif features['color_variance'] > thresholds['color_variance_empty_min'] - 200:
        score_empty += feature_weights['color_variance'] * 0.5

    if features['edge_density'] > thresholds['edge_density_empty_min']:
        score_empty += feature_weights['edge_density']
    elif features['edge_density'] > thresholds['edge_density_empty_min'] - 0.04:
        score_empty += feature_weights['edge_density'] * 0.5

    if features['dark_pixel_ratio'] < thresholds['dark_pixel_ratio_empty_max']:
        score_empty += feature_weights['dark_pixel_ratio']
    elif features['dark_pixel_ratio'] < thresholds['dark_pixel_ratio_empty_max'] + 0.08:
        score_empty += feature_weights['dark_pixel_ratio'] * 0.5

    if features['texture_complexity'] > 0.60:
        score_empty += feature_weights['texture_complexity']

    
    # 1. Couleur dominante sombre + contours élevés => pleine
    avg_color = (analysis['avg_color_r'] + analysis['avg_color_g'] + analysis['avg_color_b']) / 3
    if avg_color < 90 and features['edge_density'] > 0.15:
        score_full += 0.10

    # 2. Texture ET variance élevées => pleine
    if features['texture_complexity'] > 0.5 and features['color_variance'] > 1200:
        score_full += 0.10

    # 3. Texture ET variance faibles => vide ou fermée
    if features['texture_complexity'] < 0.2 and features['color_variance'] < 600:
        score_empty += 0.08

    # 4. Luminosité faible ET beaucoup de pixels sombres => pleine
    if features['brightness'] < 110 and features['dark_pixel_ratio'] > 0.25:
        score_full += 0.10

    # 5. Luminosité forte ET peu de pixels sombres => vide
    if features['brightness'] > 140 and features['dark_pixel_ratio'] < 0.10:
        score_empty += 0.10

    # 6. Entropie très faible => vide ou fermée
    if features['color_entropy'] < 4.0:
        score_empty += 0.08

    # 7. Entropie très élevée => pleine
    if features['color_entropy'] > 6.0:
        score_full += 0.08

    # 8. Distribution spatiale élevée => pleine
    if 'spatial_distribution' in analysis and analysis['spatial_distribution'] > 0.18:
        score_full += 0.07

    # 9. Distribution spatiale très faible => vide ou fermée
    if 'spatial_distribution' in analysis and analysis['spatial_distribution'] < 0.06:
        score_empty += 0.07

    
    # Bonus de cohérence
    indicators_full = 0
    indicators_empty = 0
    if features['brightness'] < thresholds['brightness_full_max']:
        indicators_full += 1
    if features['contrast'] < thresholds['contrast_full_max']:
        indicators_full += 1
    if features['dark_pixel_ratio'] > thresholds['dark_pixel_ratio_full_min']:
        indicators_full += 1
    if features['color_variance'] < thresholds['color_variance_full_max']:
        indicators_full += 1
    if features['brightness'] > thresholds['brightness_empty_min']:
        indicators_empty += 1
    if features['contrast'] > thresholds['contrast_empty_min']:
        indicators_empty += 1
    if features['dark_pixel_ratio'] < thresholds['dark_pixel_ratio_empty_max']:
        indicators_empty += 1
    if features['color_variance'] > thresholds['color_variance_empty_min']:
        indicators_empty += 1
    if indicators_full >= 4:
        score_full += 0.18
    if indicators_empty >= 4:
        score_empty += 0.18

    # Normalisation et calcul de confiance
    total_weight = sum(feature_weights.values()) + 0.18  # +0.18 pour les bonus
    score_full = min(0.95, score_full / total_weight)
    score_empty = min(0.95, score_empty / total_weight)

    # Décision avec seuil de confiance minimum plus bas
    if score_full > score_empty and score_full > 0.12:
        confidence = score_full + (score_full - score_empty) * 0.3
        return 'full', min(0.95, confidence)
    elif score_empty > score_full and score_empty > 0.18:
        confidence = score_empty + (score_empty - score_full) * 0.3
        return 'empty', min(0.95, confidence)
    else:
        return 'unknown', max(score_full, score_empty)

import glob

from flask import current_app
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from app.models import KNN_MODEL_FILE, RF_MODEL_FILE, SVM_MODEL_FILE

def train_all_train_folder_ml():
    X, y = [], []

    base = current_app.config['TRAINING_FOLDER']
    clean_dir = os.path.join(base, 'with_label', 'clean')
    dirty_dir = os.path.join(base, 'with_label', 'dirty')

    for img_file in glob.glob(os.path.join(clean_dir, '*')):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            feats = extract_features_for_knn(img_file)
            if feats is not None and isinstance(feats, (list, np.ndarray)):
                X.append(feats)
                y.append(0)  # label 0 = empty/clean
            else:
                current_app.logger.warning(f"Ignorée (clean) : {img_file}")

    for img_file in glob.glob(os.path.join(dirty_dir, '*')):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            feats = extract_features_for_knn(img_file)
            if feats is not None and isinstance(feats, (list, np.ndarray)):
                X.append(feats)
                y.append(1)  # label 1 = full/dirty
            else:
                current_app.logger.warning(f"Ignorée (dirty) : {img_file}")

    if not X:
        current_app.logger.error("Aucune donnée pour entraîner les modèles ML.")
        return False

    expected_len = len(X[0])
    X_clean, y_clean = [], []
    for feats, label in zip(X, y):
        arr = np.asarray(feats).flatten()
        if arr.shape[0] == expected_len:
            X_clean.append(arr)
            y_clean.append(label)
        else:
            current_app.logger.warning(
                f"Vector incorrect ({arr.shape[0]} vs {expected_len}) pour un fichier de training."
            )

    X = np.array(X_clean)
    y = np.array(y_clean)
    if X.size == 0:
        current_app.logger.error("Aucune donnée exploitable après filtrage.")
        return False

    if len(np.unique(y)) < 2:
        current_app.logger.error("Besoin d'au moins 2 classes pour l'entraînement.")
        return False

    # Normalisation
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Entraînement des modèles
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_scaled, y)
    with open(KNN_MODEL_FILE, 'wb') as f:
        pickle.dump(knn, f)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_scaled, y)
    with open(RF_MODEL_FILE, 'wb') as f:
        pickle.dump(rf, f)

    svm = SVC(probability=True, kernel='rbf', random_state=42)
    svm.fit(X_scaled, y)
    with open(SVM_MODEL_FILE, 'wb') as f:
        pickle.dump(svm, f)

    # Sauvegarde du scaler
    scaler_path = os.path.join(current_app.root_path, 'scaler_ml.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    current_app.logger.info(f"Modèles ML entraînés sur {len(y)} images.")
    return True

RULES_CONFIG_FILE = 'rules_config.json'

def load_rules_config():
    if os.path.exists(RULES_CONFIG_FILE):
        with open(RULES_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def save_rules_config(config):
    with open(RULES_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
