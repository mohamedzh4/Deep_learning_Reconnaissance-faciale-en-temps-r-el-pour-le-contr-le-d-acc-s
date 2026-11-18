# Face Access Control — Version Finale

Système complet de contrôle d'accès biométrique facial en temps réel combinant détection, anti-usurpation (liveness detection), reconnaissance faciale personnalisée (ArcFace ONNX) et vote temporel multi-images.

---

## ⚙️ Architecture globale

```
Flux Vidéo Webcam (OpenCV)
         │
         ▼
[1] Détection Faciale + 5 Landmarks (InsightFace buffalo_s / buffalo_l)
         │
         ▼
[2] Évaluation Anti-usurpation / Liveness (MobileNetV3 ONNX) ──▶ Échec ? ──▶ ALERTE SPOOF + Journal DENIED
         │ (Succès)
         ▼
[3] Alignement 5 points (Transformée d'Umeyama ArcFace 112×112)
         │
         ▼
[4] Extraction d'Embedding 512-d (Modèle custom ONNX ou InsightFace)
         │
         ▼
[5] Appariement Cosinus (Galerie biométrique sécurisée avec guard model_id)
         │
         ▼
[6] Décision par Vote Glissant + Cooldown anti-rebond (AccessDecision)
         │
         ▼
[7] Déclenchement Relais Porte + Historisation SQLite (AuditLog) + Dashboard Streamlit
```

---

## 📁 Structure du Projet

```
version_final/
├── app.py                       # Dashboard Web interactif Streamlit
├── config.yaml                  # Configuration centrale (caméra, modèles, seuils)
├── requirements.txt             # Dépendances Python
├── README.md                    # Documentation complète
├── .gitignore                   # Exclusions Git
├── src/
│   ├── __init__.py              # Définition du package
│   ├── config.py                # Chargement et mise à jour de la configuration
│   ├── camera.py                # Capture vidéo multi-OS (AVFoundation, DShow, V4L2)
│   ├── align.py                 # Alignement facial 5 points (Umeyama 112×112)
│   ├── embedder.py              # Wrapper ONNX Runtime pour recognizer personnalisé
│   ├── face_engine.py           # Moteur unifié détection + reconnaissance
│   ├── liveness.py              # Anti-spoofing passif ONNX + recadrage contextuel
│   ├── gallery.py               # Gestion de la galerie avec protection model_id
│   ├── matcher.py               # Calcul de similarité cosinus et seuillage
│   ├── decision.py              # Filtrage temporel par vote majoritaire et cooldown
│   ├── door.py                  # Contrôleur temporisé de gâche électrique / relais
│   ├── audit_log.py             # Journalisation des événements dans SQLite
│   └── pipeline.py              # Pipeline complet de bout en bout + overlay vidéo
├── scripts/
│   ├── run_realtime.py          # Exécution CLI temps réel avec worker asynchrone
│   ├── enroll_user.py           # Enrôlement interactif par webcam
│   ├── enroll_from_folders.py   # Enrôlement par lots à partir de dossiers d'images
│   ├── calibrate_threshold.py   # Analyse ROC / FAR / FRR et calibration du seuil
│   ├── collect_liveness_data.py # Collecte de données réelles et attaques pour liveness
│   └── benchmark.py             # Mesure des latences et du FPS
├── notebooks/
│   └── liveness_training.ipynb  # Notebook Colab GPU pour entraîner le modèle anti-spoof
├── tests/
│   ├── test_core.py             # Tests unitaires pipeline, galerie, matcher, décision
│   ├── test_liveness.py         # Tests unitaires recadrage et inférence liveness
│   └── test_custom_embedder.py  # Tests unitaires alignement, embedder et model_id
├── models/
│   ├── buffalo_s/               # Poids pré-entraînés InsightFace
│   ├── recognition/             # Modèles personnalisés (ex: my_model.onnx)
│   └── liveness/                # Modèle anti-spoofing (antispoof.onnx)
├── data/
│   ├── enrollment/              # Fichiers de gabarits (gallery.npz)
│   ├── own_dataset/             # Clichés d'enrôlement et sondes de test
│   ├── liveness/                # Données brutes de collecte anti-spoofing
│   └── access_log.sqlite        # Base SQLite des événements d'accès
└── evaluation/results/          # Courbes ROC et métriques exportées
```

---

## 🚀 Installation et Utilisation

### 1. Environnement Virtuel
```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate.bat    # Windows
pip install -r requirements.txt
```

### 2. Lancement du Dashboard (Recommandé)
```bash
streamlit run app.py
```
Le dashboard offre :
- **Dashboard** : statistiques journalières, KPI et graphiques d'activité.
- **Gallery** : enrôlement direct via la webcam (avec validation de pose et qualité).
- **Live Control** : flux vidéo en direct avec superposition visuelle (boxes, FPS, état porte, liveness).
- **Audit Log** : consultation filtrable de l'historique et export CSV.
- **Settings** : configuration dynamique des modèles, seuils, et activation du liveness.

### 3. Enrôlement en Ligne de Commande
```bash
python scripts/enroll_user.py --name "zahar" --n 15
```

### 4. Calibration du Seuil Biométrique
```bash
python scripts/calibrate_threshold.py --far 0.01 --write
```

### 5. Collecte et Entraînement Liveness (Phase 2)
```bash
# Collecte des sessions réelles (live)
python scripts/collect_liveness_data.py --label live --group session1

# Collecte des attaques (spoof)
python scripts/collect_liveness_data.py --label spoof --group phone_photo1
python scripts/collect_liveness_data.py --label spoof --group print_photo1
```
Entraînez ensuite sur Google Colab avec `notebooks/liveness_training.ipynb` puis placez `antispoof.onnx` et `antispoof.json` dans `models/liveness/`. Activez l'option dans **Settings** ou `config.yaml` (`liveness.enabled: true`).

