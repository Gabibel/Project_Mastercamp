from flask import Blueprint, jsonify, url_for, current_app
from app.models import TrashImage

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/stats')
def api_stats():
    data = []
    for img in TrashImage.query.all():
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

@api_bp.route('/map_data')
def api_map_data():
    data = []
    for img in TrashImage.query.all():
        data.append({
            'id': img.id,
            'latitude': img.latitude,
            'longitude': img.longitude,
            'status': img.status,
            'ai_prediction': img.ai_prediction,
            'upload_date': img.upload_date.strftime('%Y-%m-%d %H:%M'),
            'location_name': img.location_name,
            'filename': img.filename,
            'url': url_for('main.uploaded_file', filename=img.filename)
        })
    return jsonify(data)
