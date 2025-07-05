import os
import glob
import pickle
import threading

import numpy as np
from flask import current_app, flash, url_for, redirect
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from app.analysis import analyze_image_advanced, predict_with_advanced_ai, save_rules_config
from app.models import TrashImage, KNN_MODEL_FILE, RF_MODEL_FILE, SVM_MODEL_FILE
from app import db


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


def async_analyze_and_update(trash_image_id, filepath, app):
    def worker():
        with app.app_context():
            img = TrashImage.query.get(trash_image_id)
            if not img:
                return

            analysis = analyze_image_advanced(filepath)
            if not analysis:
                return

            # prédiction par règles
            rule_pred, rule_conf = predict_with_advanced_ai(analysis)

            # extraction ML
            feats = [
                analysis['brightness'],
                analysis['contrast'],
                analysis['color_variance'],
                analysis['edge_density'],
                analysis['texture_complexity'],
                analysis['dark_pixel_ratio'],
                analysis['color_entropy']
            ]

            knn_pred = rf_pred = svm_pred = None
            knn_conf = rf_conf = svm_conf = None

            # chargéé du scaler
            scaler_path = os.path.join(app.root_path, 'scaler_ml.pkl')
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            feats_scaled = scaler.transform([feats]) if scaler else [feats]

            # KNN
            if os.path.exists(KNN_MODEL_FILE):
                with open(KNN_MODEL_FILE, 'rb') as f:
                    knn = pickle.load(f)
                proba = knn.predict_proba(feats_scaled)[0]
                p = knn.predict(feats_scaled)[0]
                knn_pred, knn_conf = ('empty' if p==0 else 'full', float(proba.max()))

            # RF
            if os.path.exists(RF_MODEL_FILE):
                with open(RF_MODEL_FILE, 'rb') as f:
                    rf = pickle.load(f)
                proba = rf.predict_proba(feats_scaled)[0]
                p = rf.predict(feats_scaled)[0]
                rf_pred, rf_conf = ('empty' if p==0 else 'full', float(proba.max()))

            # SVM
            if os.path.exists(SVM_MODEL_FILE):
                with open(SVM_MODEL_FILE, 'rb') as f:
                    svm = pickle.load(f)
                proba = svm.predict_proba(feats_scaled)[0]
                p = svm.predict(feats_scaled)[0]
                svm_pred, svm_conf = ('empty' if p==0 else 'full', float(proba.max()))

            # vote majoritaire
            votes = [v for v in (knn_pred, rf_pred, svm_pred) if v in ('full','empty')]
            ml_vote = max(set(votes), key=votes.count) if votes else None

            # mise à jour
            for k, v in analysis.items():
                setattr(img, k, v)
            img.ai_prediction, img.ai_confidence = rule_pred, rule_conf
            img.knn_prediction, img.knn_confidence = knn_pred, knn_conf
            img.rf_prediction, img.rf_confidence   = rf_pred, rf_conf
            img.svm_prediction, img.svm_confidence = svm_pred, svm_conf
            img.ml_vote = ml_vote
            img.status  = 'pending'
            db.session.commit()

    # on démarre le thread
    threading.Thread(target=worker, daemon=True).start()

def train_all_train_folder_ml():
    X, y = [], []
    base = os.path.join(current_app.root_path, current_app.config['TRAINING_FOLDER'])
    for label, folder in [(0, 'clean'), (1, 'dirty')]:
        path = os.path.join(base, 'with_label', folder)
        for imgf in glob.glob(os.path.join(path, '*')):
            feats = extract_features_for_knn(imgf)
            if feats:
                X.append(feats); y.append(label)
    if not X:
        current_app.logger.warning("Pas de données de training.")
        return False

    X = np.array(X)
    y = np.array(y)
    if len(np.unique(y)) < 2:
        current_app.logger.warning("Besoin des deux classes pour entraîner.")
        return False

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=10).fit(Xs, y)
    pickle.dump(knn, open(KNN_MODEL_FILE, 'wb'))
    # RF
    rf  = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xs, y)
    pickle.dump(rf, open(RF_MODEL_FILE, 'wb'))
    # SVM
    svm = SVC(probability=True, kernel='rbf', random_state=42).fit(Xs, y)
    pickle.dump(svm, open(SVM_MODEL_FILE, 'wb'))

    pickle.dump(scaler, open(os.path.join(current_app.root_path, 'scaler_ml.pkl'), 'wb'))
    return True

def reset_rules_knn():
    import numpy as np
    base = current_app.config['TRAINING_FOLDER']
    def gather(folder): 
        feats = []
        for f in glob.glob(os.path.join(base, 'with_label', folder, '*')):
            v = extract_features_for_knn(f)
            if v: feats.append(v)
        return np.array(feats)
    clean, dirty = gather('clean'), gather('dirty')
    if clean.size == 0 or dirty.size == 0:
        flash("Pas assez de données pour reset_rules_knn.")
        return redirect(url_for('rules_bp.rules'))

    thresholds = {
        'brightness_full_max': float(dirty[:,0].max()),
        'brightness_empty_min': float(clean[:,0].min()),
        'contrast_full_max':    float(dirty[:,1].max()),
        'contrast_empty_min':   float(clean[:,1].min()),
        'color_variance_full_max': float(dirty[:,2].max()),
        'color_variance_empty_min': float(clean[:,2].min()),
        'edge_density_full_max':   float(dirty[:,3].max()),
        'edge_density_empty_min':  float(clean[:,3].min()),
        'dark_pixel_ratio_full_min': float(dirty[:,5].min()),
        'dark_pixel_ratio_empty_max': float(clean[:,5].max()),
    }
    save_rules_config(thresholds)
    flash("Règles reset_knn effectuées.")
    return redirect(url_for('rules_bp.rules'))

def resimuler():
    count = 0
    for img in TrashImage.query.all():
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], img.filename)
        if os.path.exists(path):
            async_analyze_and_update(img.id, path)
            count += 1
    flash(f"Re-simulation lancée sur {count} images.")
    return redirect(url_for('rules_bp.rules'))

def resimuler_image(image_id):
    img = TrashImage.query.get_or_404(image_id)
    path = os.path.join(current_app.config['UPLOAD_FOLDER'], img.filename)
    if os.path.exists(path):
        async_analyze_and_update(img.id, path)
        flash("Re-simulation pour une image lancée.")
    else:
        flash("Fichier introuvable.")
    return redirect(url_for('main_bp.gallery'))
