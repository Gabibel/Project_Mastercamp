from flask import Blueprint, jsonify, url_for, current_app, request
from app.models import TrashImage
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime

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
    status_filter = request.args.get('status', 'full')
    date_filter = request.args.get('date', None)

    # Requête initiale
    query = TrashImage.query
    if status_filter != "Tous":
        query = query.filter_by(status=status_filter)

    if date_filter:
        try:
            parsed_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
            query = query.filter(TrashImage.upload_date >= parsed_date)
        except ValueError:
            pass

    images = TrashImage.query.filter(TrashImage.latitude != None, TrashImage.longitude != None).all()

    images_to_cluster = [img for img in images if img.status == "full"]
    coords = [[np.radians(img.latitude), np.radians(img.longitude)] for img in images_to_cluster]
    img_ids = [img.id for img in images_to_cluster]

    labels = [-1] * len(images)
    if coords:
        db = DBSCAN(eps=2 / 6371.0, min_samples=3, metric='haversine')
        cluster_labels = db.fit_predict(coords)
        id_to_cluster = {img_id: int(label) if label != -1 else None for img_id, label in zip(img_ids, cluster_labels)}
    else:
        id_to_cluster = {}
    result = []
    for img in images:
        result.append({
            'id': img.id,
            'latitude': img.latitude,
            'longitude': img.longitude,
            'status': img.status,
            'upload_date': img.upload_date.strftime('%Y-%m-%d'),
            'original_filename': img.original_filename,
            'location_name': img.location_name,
            'brightness': img.brightness,
            'cluster_id': id_to_cluster.get(img.id)
        })
    return jsonify(result)


@api_bp.route('/clusters')
def api_clusters():
    # On clusterise uniquement les poubelles "pleines", peu importe le filtre côté client
    query = TrashImage.query.filter(
        TrashImage.status == 'full',
        TrashImage.latitude != None,
        TrashImage.longitude != None
    )

    date_filter = request.args.get('date', None)
    if date_filter:
        try:
            parsed_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
            query = query.filter(TrashImage.upload_date >= parsed_date)
        except ValueError:
            pass


    if date_filter:
        try:
            parsed_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
            query = query.filter(TrashImage.upload_date >= parsed_date)
        except ValueError:
            pass

    images = query.all()

    coords_rad = []
    for img in images:
        coords_rad.append([np.radians(img.latitude), np.radians(img.longitude)])

    if not coords_rad:
        return jsonify([])

    coords_rad = np.array(coords_rad)
    db = DBSCAN(eps=2 / 6371.0, min_samples=3, metric='haversine').fit(coords_rad)
    labels = db.labels_

    clusters = {}
    for label, coord in zip(labels, coords_rad):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(coord)

    result = []
    for cluster_points in clusters.values():
        cluster_np = np.array(cluster_points)
        mean_lat_deg = float(np.degrees(np.mean(cluster_np[:, 0])))
        mean_lon_deg = float(np.degrees(np.mean(cluster_np[:, 1])))
        result.append({
            "lat": mean_lat_deg,
            "lon": mean_lon_deg,
            "count": len(cluster_points)
        })

    return jsonify(result)

