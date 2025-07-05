from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from app.analysis import load_rules_config, save_rules_config
from app.utils import train_all_train_folder_ml, reset_rules, resimuler, resimuler_image

rules_bp = Blueprint('rules', __name__, url_prefix='/rules')

@rules_bp.route('', methods=['GET', 'POST'])
def rules():
    default_thresholds = {
        'brightness_full_max': 130, 'brightness_empty_min': 120,
        'contrast_full_max': 35,   'contrast_empty_min': 30,
        'color_variance_full_max': 900, 'color_variance_empty_min': 900,
        'edge_density_full_max': 0.18,  'edge_density_empty_min': 0.16,
        'dark_pixel_ratio_full_min': 0.18, 'dark_pixel_ratio_empty_max': 0.22
    }
    thresholds = load_rules_config() or default_thresholds

    if request.method == 'POST':
        for key in thresholds:
            val = request.form.get(key)
            if val is not None:
                try:
                    thresholds[key] = float(val)
                except ValueError:
                    pass
        save_rules_config(thresholds)
        flash('Règles sauvegardées avec succès.')
        return redirect(url_for('rules.rules'))

    return render_template('rules.html', thresholds=thresholds)

@rules_bp.route('/train_all_validated_ml')
def train_all_validated_ml():
    success = train_all_train_folder_ml()
    msg = 'Modèles ML entraînés avec succès.' if success else 'Erreur lors de l\'entraînement ML.'
    flash(msg)
    return redirect(url_for('rules.rules'))

@rules_bp.route('/reset_rules')
def reset_rules_route():
    return reset_rules()

@rules_bp.route('/resimuler')
def resimuler_route():
    return resimuler()

@rules_bp.route('/resimuler_image/<int:image_id>')
def resimuler_image_route(image_id):
    return resimuler_image(image_id)

