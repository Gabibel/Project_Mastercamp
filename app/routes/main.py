import os
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import (
    Blueprint, render_template, request,
    redirect, url_for, flash, send_file, current_app
)
from app import db
from app.models import TrashImage
from app.utils import async_analyze_and_update

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    total_images   = TrashImage.query.count()
    full_images    = TrashImage.query.filter_by(status='full').count()
    empty_images   = TrashImage.query.filter_by(status='empty').count()
    pending_images = TrashImage.query.filter_by(status='pending').count()

    today_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    recent_images = TrashImage.query.filter(
        TrashImage.upload_date >= today_utc
    ).count()

    return render_template('index.html',
        total_images=total_images,
        full_images=full_images,
        empty_images=empty_images,
        pending_images=pending_images,
        recent_images=recent_images
    )

from PIL import Image
import threading

@main_bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Aucun fichier sélectionné')
            return redirect(request.url)

        files = request.files.getlist('file')
        if not files or all(f.filename == '' for f in files):
            flash('Aucun fichier sélectionné')
            return redirect(request.url)

        # Vérification des champs obligatoires
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)
        location_name = request.form.get('location_name')
        photo_datetime_str = request.form.get('photo_datetime')

        missing_fields = []
        if latitude is None:
            missing_fields.append("Latitude")
        if longitude is None:
            missing_fields.append("Longitude")
        if not location_name:
            missing_fields.append("Nom du lieu")
        if not photo_datetime_str:
            missing_fields.append("Date/heure de la photo")

        if missing_fields:
            flash("Champs obligatoires manquants : " + ", ".join(missing_fields))
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
            upload_folder = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'])
            filepath      = os.path.join(upload_folder, filename)
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
            # Conversion de la date/heure
            try:
                photo_datetime = datetime.strptime(photo_datetime_str, '%Y-%m-%dT%H:%M')
                photo_datetime = photo_datetime.replace(tzinfo=timezone.utc)
            except Exception as e:
                print(f"Erreur parsing date/heure : {e}")
                photo_datetime = datetime.now(timezone.utc)
            # Création entrée TrashImage (status pending, sans features)
            trash_image = TrashImage(
                filename=filename,
                original_filename=file.filename,
                status='pending',
                latitude=latitude,
                longitude=longitude,
                location_name=location_name,
                upload_date=photo_datetime,
            )
            db.session.add(trash_image)
            db.session.commit()
            app = current_app._get_current_object()
            threading.Thread(target=async_analyze_and_update,args=(trash_image.id, filepath, app),daemon=True).start()
            results.append(f"<b>{file.filename}</b> : uploadé, analyse en cours...")
        if results:
            flash('Résultats upload :<br>' + '<br>'.join(results))
        else:
            flash('Aucune image valide uploadée')
        return redirect(url_for('main.gallery'))
    return render_template('upload.html')

@main_bp.route('/gallery')
def gallery():
    status = request.args.get('status','all')
    page   = request.args.get('page', 1, type=int)

    q = TrashImage.query
    if   status == 'unknown':
        q = q.filter_by(ai_prediction='unknown')
    elif status != 'all':
        q = q.filter_by(status=status)

    images = q.order_by(TrashImage.upload_date.desc())\
              .paginate(page=page, per_page=12, error_out=False)

    return render_template('gallery.html',
        images=images, status_filter=status
    )

@main_bp.route('/validate/<int:image_id>', methods=['POST'])
def validate_prediction(image_id):
    img = TrashImage.query.get_or_404(image_id)
    user_status = request.form.get('status')

    if user_status in ['full','empty']:
        img.status        = user_status
        img.manual_status = user_status
        img.ai_validated  = True
        img.ai_correct    = (img.ai_prediction == user_status)
        img.ml_correct    = (img.ml_vote       == user_status)
        db.session.commit()

        if img.ml_correct:
            flash(f'✅ Vote ML correct ! ({img.ml_vote})')
        else:
            flash(f'❌ Vote ML incorrect. Vote : {img.ml_vote}, Réel : {user_status}')

    else:
        flash('Statut invalide')

    return redirect(url_for('main.gallery'))

@main_bp.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    img = TrashImage.query.get_or_404(image_id)
    upload_folder = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'])
    path = os.path.join(upload_folder, img.filename)
    if os.path.exists(path):
        os.remove(path)
    db.session.delete(img)
    db.session.commit()
    flash("Image supprimée avec succès.")
    return redirect(url_for('main.gallery'))

from flask import send_from_directory

@main_bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    # Envoie simplement le fichier depuis UPLOAD_FOLDER
    return send_from_directory(
        current_app.config['UPLOAD_FOLDER'],
        filename
    )

import os
import pickle
import numpy as np

from flask import (
    Blueprint, jsonify, current_app
)
from app.models import TrashImage, KNN_MODEL_FILE, RF_MODEL_FILE, SVM_MODEL_FILE
from app.utils import analyze_image_advanced, predict_with_advanced_ai

@main_bp.route('/resimuler_image_ajax/<int:image_id>', methods=['POST'])
def resimuler_image_ajax(image_id):
    img = TrashImage.query.get_or_404(image_id)
    upload_folder = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'])
    filepath      = os.path.join(upload_folder, img.filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Fichier introuvable'}), 404
    print("test")
    analysis = analyze_image_advanced(filepath)
    if not analysis:
        return jsonify({'success': False, 'error': 'Analyse impossible'}), 500

    # IA par règles
    rule_pred, rule_conf = predict_with_advanced_ai(analysis)

    # Extraction des features pour ML
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

    try:
        scaler = None
        scaler_path = os.path.join(current_app.root_path, 'scaler_ml.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
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
        current_app.logger.error(f"Erreur prédiction ML (ajax) : {e}")

    # Vote majoritaire
    votes = [p for p in (knn_pred, rf_pred, svm_pred) if p in ('full','empty')]
    ml_vote = max(set(votes), key=votes.count) if votes else None

    # Mise à jour de l'objet
    for k, v in analysis.items():
        setattr(img, k, v)
    img.ai_prediction   = rule_pred
    img.ai_confidence   = rule_conf
    img.knn_prediction  = knn_pred
    img.knn_confidence  = knn_conf
    img.rf_prediction   = rf_pred
    img.rf_confidence   = rf_conf
    img.svm_prediction  = svm_pred
    img.svm_confidence  = svm_conf
    img.ml_vote         = ml_vote
    img.status          = 'pending'
    db.session.commit()

    # Préparer le badge HTML
    if img.status == 'full':
        badge = '<span class="badge bg-danger"><i class="fas fa-exclamation-triangle me-1"></i>Pleine</span>'
    elif img.status == 'empty':
        badge = '<span class="badge bg-success"><i class="fas fa-check-circle me-1"></i>Vide</span>'
    elif img.ai_prediction == 'unknown':
        badge = '<span class="badge bg-secondary"><i class="fas fa-question-circle me-1"></i>Incertaine</span>'
    else:
        badge = '<span class="badge bg-warning text-dark"><i class="fas fa-clock me-1"></i>En attente</span>'

    ia_status = ''
    if img.ai_validated:
        ia_status = (
            '<small class="text-success"><i class="fas fa-check-circle me-1"></i>IA correcte</small>'
            if img.ml_correct
            else '<small class="text-danger"><i class="fas fa-times-circle me-1"></i>IA incorrecte</small>'
        )

    return jsonify({
        'success': True,
        'badge_html': badge,
        'ia_status_html': ia_status,
        'ml_vote': img.ml_vote,
        'ai_prediction': img.ai_prediction,
        'ai_confidence': img.ai_confidence,
        'knn_prediction': img.knn_prediction,
        'rf_prediction': img.rf_prediction,
        'svm_prediction': img.svm_prediction
    })

@main_bp.route('/validate_ajax/<int:image_id>', methods=['POST'])
def validate_prediction_ajax(image_id):
    image = TrashImage.query.get_or_404(image_id)
    user_status = request.form.get('status')
    if user_status in ['full', 'empty']:
        image.status = user_status
        image.manual_status = user_status
        image.ai_validated = True
        image.ai_correct = (image.ai_prediction == user_status)
        image.ml_correct = (image.ml_vote == user_status)
        db.session.commit()
        # Préparer la réponse JSON pour mise à jour dynamique
        badge = ''
        if user_status == 'full':
            badge = '<span class="badge bg-danger"><i class="fas fa-exclamation-triangle me-1"></i>Pleine</span>'
        elif user_status == 'empty':
            badge = '<span class="badge bg-success"><i class="fas fa-check-circle me-1"></i>Vide</span>'
        ia_status = ''
        if image.ml_correct:
            ia_status = '<small class="text-success"><i class="fas fa-check-circle me-1"></i>IA correcte</small>'
        else:
            ia_status = '<small class="text-danger"><i class="fas fa-times-circle me-1"></i>IA incorrecte</small>'
        return jsonify({
            'success': True,
            'status': user_status,
            'badge_html': badge,
            'ia_status_html': ia_status,
            'ml_vote': image.ml_vote,
            'ai_prediction': image.ai_prediction,
            'ai_confidence': image.ai_confidence,
            'knn_prediction': image.knn_prediction,
            'rf_prediction': image.rf_prediction,
            'svm_prediction': image.svm_prediction
        })
    else:
        return jsonify({'success': False, 'error': 'Statut invalide'})



from flask import render_template
from app.models import TrashImage
import numpy as np

@main_bp.route('/audit')
def audit():
    images = TrashImage.query.order_by(TrashImage.upload_date.desc()).all()
    audit_results = []
    for img in images:
        anomalies = []
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], img.filename)
        if not os.path.exists(filepath):
            anomalies.append("Fichier manquant")
        # … le reste de votre détection d’anomalies …
        audit_results.append({"image": img, "anomalies": anomalies})

    # calcul des stats des features par classe (full/empty)
    stats = {}
    features = ["brightness", "contrast", "color_variance", "edge_density", "texture_complexity", "dark_pixel_ratio", "color_entropy"]
    for classe in ["full", "empty"]:
        classe_imgs = [i for i in images if i.status == classe]
        stats[classe] = {}
        for feat in features:
            vals = [getattr(i, feat) for i in classe_imgs if getattr(i, feat) is not None]
            stats[classe][feat] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std":  float(np.std(vals))  if vals else None,
                "count": len(vals)
            }

    return render_template('audit.html',
                            audit_results=audit_results,
                            feature_stats=stats)
