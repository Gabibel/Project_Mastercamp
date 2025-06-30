# 🗑️ SmartTrash Monitor - Plateforme intelligente de suivi des poubelles

## 📋 Description

SmartTrash Monitor est une plateforme web intelligente de suivi de l'état des poubelles urbaines, conçue pour la prévention des dépôts sauvages. Elle permet l'upload d'images, l'analyse automatique de l'état des poubelles (pleine/vide), l'annotation manuelle, la visualisation des données, la cartographie dynamique et l'audit de la qualité des données.

---

## ✅ Fonctionnalités présentes dans le projet

- **Upload d'images** (citoyen/agent/caméra) avec stockage physique
- **Compression automatique** des images à l'upload (JPEG, max 1280px)
- **Extraction automatique de caractéristiques** (taille, dimensions, couleur, contraste, contours, texture, entropie, etc.)
- **Base de données structurée** (SQLite via SQLAlchemy)
- **Annotation manuelle** (pleine/vide) via l'interface
- **Classification automatique** (règles conditionnelles configurables via interface)
- **Tableau de bord** avec :
  - Statistiques globales (nombre d'images, répartition, etc.)
  - **Graphiques dynamiques interactifs avec Chart.js** (répartition, performance IA, évolution de la luminosité, etc.)
  - **Filtres dynamiques** (par statut, par date)
  - **Comparaison des modèles IA** (KNN, RF, SVM, règles)
- **Cartographie dynamique** avec **Leaflet.js** (affichage des poubelles sur une carte, filtres)
- **API REST** pour les statistiques et la carte (`/api/stats`, `/api/map_data`)
- **Audit automatique** de la base (anomalies, cohérence, page dédiée)
- **Pagination** de la galerie
- **Traitement asynchrone** de l'analyse d'images (threading)
- **Interface responsive** (Bootstrap)
- **Sécurité upload** (formats, taille, nom sécurisé)
- **Logs d'erreur** en console

---

## 🚧 Fonctionnalités à développer/améliorer

- **Gestion multilingue** (français/anglais)
- **Explicabilité avancée** des décisions IA/ML (affichage détaillé des règles appliquées et des scores pour chaque image)
- **Filtres avancés supplémentaires** (par localisation, plage de dates, etc.)
- **Section Green IT & évaluation des risques** enrichie (questionnaire, documentation dédiée)
- **Documentation technique complète** (structure, captures d'écran, démarche Green IT, évaluation des risques, script de démo)
- **(Optionnel/Bonus)** : Intégration de données contextuelles externes (météo, population, etc.)

---

## 🛠️ Technologies Utilisées

- **Backend** : Python, Flask, SQLAlchemy, SQLite, Pillow, OpenCV, NumPy, Matplotlib
- **Frontend** : HTML/CSS, Bootstrap, JavaScript, Chart.js, Leaflet.js

## 📦 Installation

1. Cloner le projet
2. Créer un environnement virtuel et installer les dépendances (`pip install -r requirements.txt`)
3. Créer les dossiers nécessaires (`uploads`, `training_data/with_label/clean`, etc.)
4. Lancer l'application (`python app.py`)
5. Accéder à http://localhost:5000

## 🎮 Utilisation

- **Upload** : Page d'upload, sélection d'images, compression et analyse automatique
- **Annotation** : Galerie, annotation manuelle, validation/correction des prédictions
- **Dashboard** : Statistiques globales, graphiques dynamiques (Chart.js), filtres, comparaison IA, carte interactive (Leaflet)
- **Audit** : Vérification de la cohérence des données
- **Configuration des règles** : Interface web pour ajuster les seuils de classification

## 🏗️ Architecture

- `app.py` : Application principale Flask
- `requirements.txt` : Dépendances Python
- `rules_config.json` : Seuils de classification
- `instance/trash_monitoring.db` : Base SQLite
- `uploads/` : Images uploadées
- `training_data/with_label/clean|dirty` : Données d'entraînement
- `templates/` : Templates HTML (Bootstrap)

## 🔧 Configuration

- Variables d'environnement dans `app.py`
- Seuils de classification dans `rules_config.json` (modifiables via l'interface)

## 📈 API Endpoints

- `/api/stats` : Statistiques images (JSON)
- `/api/map_data` : Données cartographiques (JSON)
- `/static-graph/luminosity|status|contrast` : Graphes statiques (PNG)

## 🎯 Système de Classification

- Extraction de caractéristiques visuelles (luminosité, contraste, variance, etc.)
- Application de règles pondérées (configurables)
- Prédiction automatique (pleine/vide/unknown) avec score de confiance
- Annotation manuelle pour validation/correction

## 🔍 Audit et Qualité des Données

- Vérification automatique : fichiers manquants, dimensions invalides, statuts incohérents, etc.
- Statistiques par classe (pleine/vide)

## 🚀 Optimisations

- Compression d'images à l'upload (JPEG, max 1280px)
- Traitement asynchrone (threading)
- Pagination

## 🎨 Interface Utilisateur

- Design responsive (Bootstrap)
- Navigation claire, modals, raccourcis clavier
- Affichage des métadonnées et des prédictions
- Graphiques dynamiques (Chart.js)
- Carte interactive (Leaflet)

## 🔒 Sécurité

- Validation des fichiers (format, taille, nom sécurisé)
- Protection BDD (SQLAlchemy)

## 🐛 Débogage et Maintenance

- Logs d'erreur en console
- Réinitialisation facile de la base

## 📚 Dépendances

Voir `requirements.txt` pour le backend. Frontend : Bootstrap, Chart.js, Leaflet.

## 🌱 Démarche Green IT & Évaluation des Risques

- **Green IT** : Compression d'images, traitement asynchrone, pagination pour limiter la charge serveur et la bande passante.
- **Évaluation des risques** :
  - Risques techniques (perte de données, surcharge serveur, sécurité upload)
  - Risques d'usage (mauvaise annotation, données incomplètes)
  - Limites écologiques (stockage, transport de données)
- **À faire** : Ajouter un questionnaire d'écoconception et une section dédiée dans la documentation.

## 📄 Documentation technique & Démo

- **Documentation** :
  - Structure du projet
  - Fonctionnement de l'extraction de caractéristiques
  - Logique des règles de classification
  - Captures d'écran des principales fonctionnalités
  - Évaluation des risques et démarche Green IT
- **Démo** :
  - Scénario utilisateur : upload → annotation → dashboard → audit → configuration
  - Script de présentation pour la soutenance

## 👥 Auteurs

- [Votre nom]
- Projet Mastercamp Data 2024

---

## 🏆 Axes d'Amélioration Restants

- Gestion multilingue (français/anglais)
- Explicabilité avancée des décisions IA/ML
- Filtres avancés supplémentaires
- Section Green IT et évaluation des risques enrichie
- Documentation technique et script de démo détaillés
- (Bonus) Intégration de données contextuelles externes
