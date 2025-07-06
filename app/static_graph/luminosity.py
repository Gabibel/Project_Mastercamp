from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Blueprint, send_file
from app.models import TrashImage

graph_bp = Blueprint('graph', __name__, url_prefix='/static-graph')

@graph_bp.route('/luminosity')
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

@graph_bp.route('/status')
def static_graph_status():
    images = TrashImage.query.all()
    status_labels = ['pleine', 'vide']
    status_counts = [
        sum(1 for img in images if img.ml_vote == 'full'),
        sum(1 for img in images if img.ml_vote == 'empty'),
    ]
    plt.figure(figsize=(5,5))
    plt.pie(
        status_counts,
        labels=status_labels,
        autopct='%1.1f%%',
        colors=['#e74c3c', '#2ecc71']
    )
    plt.title('Répartition des statuts (vote ML)')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype='image/png')

@graph_bp.route('/contrast')
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
