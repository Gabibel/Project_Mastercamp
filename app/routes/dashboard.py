import os
import glob
import numpy as np
from flask import Blueprint, render_template, current_app
from app import db
from app.models import TrashImage
from sklearn.metrics import precision_score, recall_score, f1_score

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard_bp.route('')
def dashboard():
    # Totaux
    total_images = TrashImage.query.count()
    full_images  = TrashImage.query.filter_by(ml_vote='full').count()
    empty_images = TrashImage.query.filter_by(ml_vote='empty').count()

    # Performance IA validée
    validated = TrashImage.query.filter_by(ai_validated=True).all()
    total_validated = len(validated)
    ml_correct   = sum(1 for img in validated if img.ml_correct)
    ml_incorrect = sum(1 for img in validated if img.ml_correct is False)
    ml_accuracy  = (ml_correct / total_validated * 100) if total_validated else 0

    # Répartition des prédictions IA (avant validation)
    all_ai = TrashImage.query.filter(TrashImage.ai_prediction.isnot(None)).all()
    ai_pred_full   = sum(1 for img in all_ai if img.ai_prediction == 'full')
    ai_pred_empty  = sum(1 for img in all_ai if img.ai_prediction == 'empty')
    ai_pred_unknown = sum(1 for img in validated if img.ai_prediction not in ['full','empty'])

    # Précision par classe (vote ML)
    full_validated  = [img for img in validated if img.ml_vote == 'full']
    empty_validated = [img for img in validated if img.ml_vote == 'empty']
    full_correct = sum(1 for img in full_validated if img.ml_correct)
    empty_correct = sum(1 for img in empty_validated if img.ml_correct)
    full_accuracy  = (full_correct  / len(full_validated) * 100) if full_validated else 0
    empty_accuracy = (empty_correct / len(empty_validated) * 100) if empty_validated else 0

    # Comparaison modèles
    def metrics(pred_attr):
        preds = [getattr(img, pred_attr) for img in validated if getattr(img, pred_attr) in ['full','empty']]
        truths = [img.manual_status         for img in validated if getattr(img, pred_attr) in ['full','empty']]
        prec = precision_score(truths, preds, pos_label='full', zero_division=0) if preds else 0
        rec  = recall_score   (truths, preds, pos_label='full', zero_division=0) if preds else 0
        f1   = f1_score       (truths, preds, pos_label='full', zero_division=0) if preds else 0
        corr = sum(1 for img in validated if getattr(img, pred_attr) == img.manual_status)
        tot  = sum(1 for img in validated if getattr(img, pred_attr) in ['full','empty'])
        return prec, rec, f1, corr, tot

    knn_precision, knn_recall, knn_f1, knn_correct, knn_total   = metrics('knn_prediction')
    rf_precision,  rf_recall,  rf_f1,  rf_correct,  rf_total    = metrics('rf_prediction')
    svm_precision, svm_recall, svm_f1, svm_correct, svm_total = metrics('svm_prediction')
    rules_precision, rules_recall, rules_f1, rules_correct, rules_total = metrics('ai_prediction')
    rules_indecis = sum(1 for img in validated if img.ai_prediction not in ['full','empty'])

    # Comptage images de training
    # Récupère le chemin absolu du dossier du projet (là où se trouve run.py)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dossier d'entraînement relatif à la racine du projet
    training_folder = os.path.join(PROJECT_ROOT, 'training_data')

    clean_dir = os.path.join(training_folder, 'with_label', 'clean')
    dirty_dir = os.path.join(training_folder, 'with_label', 'dirty')

    empty_training = len([f for f in glob.glob(f"{clean_dir}/*") if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))]) if os.path.exists(clean_dir) else 0
    full_training  = len([f for f in glob.glob(f"{dirty_dir}/*") if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))]) if os.path.exists(dirty_dir) else 0

    print("clean_dir =", clean_dir)
    print("os.path.exists(clean_dir) =", os.path.exists(clean_dir))
    print("Images trouvées :", [f for f in glob.glob(f"{clean_dir}/*") if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif'))])

    return render_template('dashboard.html',
        total_images=total_images,
        full_images=full_images,
        empty_images=empty_images,
        empty_training=empty_training,
        full_training=full_training,
        ml_correct=ml_correct,
        ml_incorrect=ml_incorrect,
        ml_accuracy=ml_accuracy,
        total_validated=total_validated,
        ai_pred_full=ai_pred_full,
        ai_pred_empty=ai_pred_empty,
        ai_pred_unknown=ai_pred_unknown,
        full_accuracy=full_accuracy,
        empty_accuracy=empty_accuracy,
        full_validated_count=len(full_validated),
        empty_validated_count=len(empty_validated),
        knn_correct=knn_correct, knn_total=knn_total,
        rf_correct=rf_correct,   rf_total=rf_total,
        svm_correct=svm_correct, svm_total=svm_total,
        rules_correct=rules_correct, rules_total=rules_total, rules_indecis=rules_indecis,
        knn_precision=knn_precision, knn_recall=knn_recall, knn_f1=knn_f1,
        rf_precision= rf_precision,  rf_recall= rf_recall,  rf_f1= rf_f1,
        svm_precision=svm_precision, svm_recall=svm_recall, svm_f1=svm_f1,
        rules_precision=rules_precision, rules_recall=rules_recall, rules_f1=rules_f1
    )
