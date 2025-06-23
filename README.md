# 🗑️ SmartTrash Monitor - Plateforme de Surveillance des Poubelles

## 📋 Description

SmartTrash Monitor est une plateforme web complète de surveillance et d'analyse des poubelles urbaines. Elle permet l'upload d'images, l'analyse automatique de l'état des poubelles (pleine/vide), l'annotation manuelle, et la visualisation des données avec des tableaux de bord interactifs.

## ✨ Fonctionnalités Principales

### 🎯 Niveau 1 - Basique (Must)
- ✅ **Plateforme web simple** : Upload, affichage et annotation d'images
- ✅ **Interface d'annotation personnalisée** : Navigation fluide entre images
- ✅ **Extraction de caractéristiques de base** : Taille, dimensions, couleur moyenne
- ✅ **Stockage en base de données** : SQLite avec SQLAlchemy
- ✅ **Règles de classification codées** : Système de règles conditionnelles (if/else)
- ✅ **Visualisation statistiques basiques** : Graphes matplotlib statiques

### 🚀 Niveau 2 - Intermédiaire (Should)
- ✅ **Interface d'annotation UX avancée** : 
  - Navigation entre images avec boutons précédent/suivant
  - Raccourcis clavier (F/E/←/→) dans le modal d'annotation
  - Affichage des métadonnées au survol
- ✅ **Extraction de caractéristiques avancées** :
  - Histogrammes de couleur
  - Contraste et luminosité
  - Densité de contours (OpenCV)
  - Complexité de texture
  - Ratio de pixels sombres
  - Entropie des couleurs
  - Distribution spatiale
- ✅ **Règles de classification configurables** : Interface web pour modifier les seuils
- ✅ **Tableau de bord interactif** : Chart.js + Leaflet pour cartographie
- ✅ **Graphes dynamiques** : Statistiques en temps réel

### 🔥 Niveau 3 - Avancé (Could Have)
- ✅ **Vérification de conformité des données** : Page d'audit avec détection d'anomalies
- ✅ **Optimisation de performance** :
  - Compression d'images à l'upload (JPEG qualité 80, max 1280px)
  - Traitement asynchrone des analyses (threads Python)
  - Gestion mémoire optimisée
- ✅ **Pagination** : Navigation par pages dans la galerie
- ✅ **API REST** : Endpoint `/api/stats` pour les données JSON

## 🛠️ Technologies Utilisées

### Backend
- **Flask** : Framework web Python
- **SQLAlchemy** : ORM pour la base de données
- **SQLite** : Base de données légère
- **PIL/Pillow** : Traitement d'images
- **OpenCV** : Analyse d'images avancée
- **NumPy** : Calculs numériques
- **Matplotlib** : Génération de graphes statiques

### Frontend
- **Bootstrap 5** : Interface responsive
- **Chart.js** : Graphes dynamiques interactifs
- **Leaflet** : Cartographie interactive
- **Font Awesome** : Icônes
- **JavaScript** : Interactions utilisateur

### Optimisations
- **Threading** : Traitement asynchrone
- **Compression d'images** : Optimisation stockage
- **Pagination** : Performance avec grandes listes

## 📦 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes d'installation

1. **Cloner le projet**
```bash
git clone <url-du-repo>
cd Project_Mastercamp
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Créer les dossiers nécessaires**
```bash
mkdir uploads
mkdir training_data
mkdir training_data/with_label
mkdir training_data/with_label/clean
mkdir training_data/with_label/dirty
mkdir training_data/no_label
```

5. **Lancer l'application**
```bash
python app.py
```

6. **Accéder à l'application**
Ouvrir http://localhost:5000 dans votre navigateur

## 🎮 Utilisation

### 📤 Upload d'Images
1. Aller sur la page **Upload**
2. Sélectionner une ou plusieurs images
3. Les images sont automatiquement :
   - Compressées (JPEG qualité 80, max 1280px)
   - Analysées en arrière-plan
   - Classifiées selon les règles configurées

### 🖼️ Galerie et Annotation
1. **Galerie** : Visualiser toutes les images uploadées
2. **Filtres** : Filtrer par statut (pleine/vide/en attente)
3. **Annotation** : 
   - Cliquer sur une image pour ouvrir le modal
   - Utiliser les boutons ou raccourcis clavier (F/E/←/→)
   - Valider ou corriger les prédictions IA

### 📊 Dashboard
- **Statistiques globales** : Nombre total d'images, répartition par statut
- **Graphes dynamiques** : Chart.js pour visualisations interactives
- **Carte interactive** : Leaflet pour géolocalisation
- **Graphes statiques** : Matplotlib pour histogrammes et camemberts

### ⚙️ Configuration des Règles
1. Aller sur **Règles** dans la navigation
2. Modifier les seuils de classification :
   - Luminosité (pleine < 110, vide > 150)
   - Contraste (pleine < 30, vide > 50)
   - Variance des couleurs
   - Densité de contours
   - Ratio de pixels sombres
3. Sauvegarder les modifications

### 🔍 Audit des Données
1. Aller sur **Audit** dans la navigation
2. Vérifier la conformité des données :
   - Fichiers manquants
   - Valeurs aberrantes
   - Statuts incohérents
   - Métadonnées manquantes

## 🏗️ Architecture

### Structure des Fichiers
```
Project_Mastercamp/
├── app.py                 # Application principale Flask
├── requirements.txt       # Dépendances Python
├── rules_config.json     # Configuration des règles de classification
├── instance/
│   └── trash_monitoring.db  # Base de données SQLite
├── uploads/              # Images uploadées
├── training_data/        # Données d'entraînement
│   ├── with_label/
│   │   ├── clean/        # Images de poubelles vides
│   │   └── dirty/        # Images de poubelles pleines
│   └── no_label/         # Images non étiquetées
└── templates/            # Templates HTML
    ├── base.html         # Template de base
    ├── index.html        # Page d'accueil
    ├── upload.html       # Page d'upload
    ├── gallery.html      # Galerie d'images
    ├── dashboard.html    # Tableau de bord
    ├── rules.html        # Configuration des règles
    └── audit.html        # Page d'audit
```

### Modèle de Données
```python
class TrashImage(db.Model):
    # Identifiants
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    original_filename = db.Column(db.String(255))
    upload_date = db.Column(db.DateTime)
    
    # Statuts
    status = db.Column(db.String(50))  # 'full', 'empty', 'pending'
    manual_status = db.Column(db.String(50))
    
    # Prédictions IA
    ai_prediction = db.Column(db.String(50))
    ai_confidence = db.Column(db.Float)
    ai_validated = db.Column(db.Boolean)
    ai_correct = db.Column(db.Boolean)
    
    # Métadonnées de base
    file_size = db.Column(db.Integer)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    avg_color_r/g/b = db.Column(db.Float)
    contrast = db.Column(db.Float)
    brightness = db.Column(db.Float)
    
    # Caractéristiques avancées
    color_variance = db.Column(db.Float)
    edge_density = db.Column(db.Float)
    texture_complexity = db.Column(db.Float)
    dark_pixel_ratio = db.Column(db.Float)
    color_entropy = db.Column(db.Float)
    spatial_distribution = db.Column(db.Float)
    
    # Géolocalisation
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    location_name = db.Column(db.String(255))
```

## 🔧 Configuration

### Variables d'Environnement
```python
app.config['SECRET_KEY'] = 'votre_cle_secrete_ici'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trash_monitoring.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRAINING_FOLDER'] = 'training_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
```

### Règles de Classification (rules_config.json)
```json
{
  "brightness_full_max": 110,
  "brightness_empty_min": 150,
  "contrast_full_max": 30,
  "contrast_empty_min": 50,
  "color_variance_full_max": 700,
  "color_variance_empty_min": 1300,
  "edge_density_full_max": 0.13,
  "edge_density_empty_min": 0.22,
  "dark_pixel_ratio_full_min": 0.35,
  "dark_pixel_ratio_empty_max": 0.12
}
```

## 📈 API Endpoints

### GET `/api/stats`
Retourne les statistiques des images au format JSON :
```json
[
  {
    "id": 1,
    "date": "2024-01-15",
    "status": "full",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "location_name": "Paris",
    "brightness": 85.2,
    "contrast": 28.1,
    "avg_color_r": 120.5,
    "avg_color_g": 110.2,
    "avg_color_b": 95.8
  }
]
```

### GET `/static-graph/luminosity`
Génère un histogramme de luminosité (PNG)

### GET `/static-graph/status`
Génère un camembert des statuts (PNG)

### GET `/static-graph/contrast`
Génère un histogramme de contraste (PNG)

## 🎯 Système de Classification

### Algorithme de Prédiction
Le système utilise un **système de règles pondérées** (pas de machine learning) :

1. **Extraction de caractéristiques** :
   - Luminosité moyenne
   - Contraste
   - Variance des couleurs
   - Densité de contours
   - Complexité de texture
   - Ratio de pixels sombres
   - Entropie des couleurs

2. **Application des règles** :
   - Chaque caractéristique a un poids
   - Comparaison avec des seuils configurables
   - Calcul d'un score de confiance

3. **Décision finale** :
   - Classe avec le score le plus élevé
   - Confiance normalisée entre 0 et 1

### Formule de Score
```python
score_full = Σ(poids_caractéristique × condition_remplie)
score_empty = Σ(poids_caractéristique × condition_remplie)
confiance = max(score_full, score_empty) + bonus_différence
```

## 🔍 Fonctionnalités d'Audit

### Vérifications Automatiques
- **Fichiers manquants** : Images référencées en base mais absentes du disque
- **Dimensions invalides** : Largeur/hauteur nulles ou négatives
- **Taille de fichier** : Fichiers vides ou corrompus
- **Métadonnées manquantes** : Caractéristiques non extraites
- **Statuts incohérents** : Valeurs non reconnues
- **Prédictions manquantes** : Analyses non effectuées

### Interface d'Audit
- Tableau avec indicateurs visuels (vert/rouge)
- Détail des anomalies par image
- Aperçu des images (si disponibles)

## 🚀 Optimisations de Performance

### Compression d'Images
- **Redimensionnement** : Max 1280px (largeur ou hauteur)
- **Format** : Conversion en JPEG qualité 80
- **Optimisation** : Compression progressive

### Traitement Asynchrone
- **Upload** : Immédiat, sans blocage
- **Analyse** : Thread séparé en arrière-plan
- **Interface** : Reste responsive pendant l'analyse

### Gestion Mémoire
- **Images** : Ouverture en mode stream
- **Base de données** : Requêtes optimisées
- **Pagination** : Chargement par pages

## 🎨 Interface Utilisateur

### Design Responsive
- **Bootstrap 5** : Compatible mobile/desktop
- **Navigation** : Barre de navigation intuitive
- **Thème** : Interface moderne et claire

### Interactions
- **Raccourcis clavier** : F (pleine), E (vide), ←/→ (navigation)
- **Modals** : Affichage en plein écran
- **Tooltips** : Informations au survol
- **Alertes** : Messages de confirmation/erreur

### Visualisations
- **Graphes dynamiques** : Chart.js interactifs
- **Carte** : Leaflet avec marqueurs
- **Graphes statiques** : Matplotlib intégrés
- **Tableaux** : Données structurées

## 🔒 Sécurité

### Validation des Fichiers
- **Extensions autorisées** : PNG, JPG, JPEG, GIF, BMP
- **Taille maximale** : 16MB par fichier
- **Noms sécurisés** : `secure_filename()` de Werkzeug

### Base de Données
- **SQLAlchemy** : Protection contre les injections SQL
- **Validation** : Contrôle des types de données
- **Transactions** : Rollback en cas d'erreur

## 🐛 Débogage et Maintenance

### Logs
- **Erreurs** : Affichage dans la console
- **Uploads** : Suivi des fichiers traités
- **Analyses** : Statut des traitements

### Base de Données
- **Réinitialisation** : `db.create_all()` au démarrage
- **Sauvegarde** : Fichier SQLite dans `instance/`
- **Migration** : Compatible avec les évolutions

### Performance
- **Monitoring** : Temps de traitement
- **Mémoire** : Gestion des ressources
- **Optimisation** : Cache et compression

## 📚 Dépendances

### Backend
```
Flask>=2.3.0
Flask-SQLAlchemy>=3.0.0
Werkzeug>=2.3.0
Pillow>=9.0.0
opencv-python>=4.5.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### Frontend (CDN)
- Bootstrap 5.3.0
- Chart.js 3.9.0
- Leaflet 1.9.0
- Font Awesome 6.0.0

## 🤝 Contribution

### Structure du Code
- **Modulaire** : Séparation claire des responsabilités
- **Documenté** : Commentaires et docstrings
- **Testable** : Fonctions isolées et réutilisables

### Ajout de Fonctionnalités
1. **Backend** : Routes Flask dans `app.py`
2. **Frontend** : Templates HTML dans `templates/`
3. **Base de données** : Modèles SQLAlchemy
4. **Tests** : Validation des nouvelles fonctionnalités

## 📄 Licence

Ce projet est développé dans le cadre du Mastercamp. Tous droits réservés.

## 👥 Auteurs

- **Développeur** : [Votre nom]
- **Projet** : SmartTrash Monitor
- **Date** : 2024

---

## 🎯 Prochaines Évolutions Possibles

### Fonctionnalités Avancées
- **Machine Learning** : Intégration de modèles CNN
- **API REST complète** : Endpoints pour intégration externe
- **Notifications** : Alertes en temps réel
- **Export de données** : CSV, Excel, PDF
- **Authentification** : Système de comptes utilisateurs
- **Multi-utilisateurs** : Gestion des rôles et permissions

### Optimisations
- **Cache Redis** : Mise en cache des analyses
- **CDN** : Distribution des images
- **Docker** : Containerisation
- **Tests automatisés** : Unit tests et intégration
- **CI/CD** : Pipeline de déploiement

### Intégrations
- **IoT** : Capteurs connectés
- **Météo** : API météorologique
- **Maps** : Intégration Google Maps/OpenStreetMap
- **Analytics** : Google Analytics, Mixpanel
- **Monitoring** : Sentry, New Relic
