# 🧠 Medical AI Agent — Analyse de Tumeurs Cérébrales par IRM

**Système Multi-Agent d’Intelligence Artificielle pour la Détection, Segmentation, Classification et Génération de Rapports Cliniques.**

Développé sur le dataset **BRISC 2025** (images IRM 2D pondérées T1).

---

## 📋 Aperçu

**Medical AI Agent** est une plateforme médicale avancée conçue pour assister les radiologues et neurochirurgiens dans l'analyse d'IRM cérébrales. Le système intègre des modèles de vision par ordinateur pour la localisation et le typage tumoral, un agent conversationnel clinique fondé sur des connaissances médicales, ainsi qu'une traçabilité réglementaire stricte.

### 🌟 Fonctionnalités clés

- **Segmentation Tumorigène** : 
  - Modèle **Attention U-Net** (MONAI) avec convolutions résiduelles et mécanismes d'attention spatiale.
  - Support de recherche et fine-tuning **nnU-Net 2D**.
- **Classification Multiclasse Dédiée** :
  - Diagnostic 4 classes : **Gliome**, **Méningiome**, **Pituitaire (adénome)**, **Pas de tumeur**.
  - Architecture : **EfficientNetV2-S** (optimisé pour une haute précision et un ratio paramètres/convergence supérieur).
- **Explicabilité Visuelle (XAI)** :
  - Cartes d'activation **Grad-CAM** superposées à l'IRM pour identifier les zones cérébrales influençant la décision du classifieur.
  - Visualisations comparatives (Image d'origine, Ground Truth, Masque prédit, Overlay Grad-CAM).
- **Agent Dialogue Clinique & LLM Quantifié (4-bit)** :
  - Intégration d'un LLM médical conversationnel quantifié en 4-bit (NF4).
  - Génération sans contrainte rigide de tokens (`max_length` aligné sur le contexte natif jusqu'à 128k).
  - Gestion de sessions de conversation multi-tours isolées (`session_id`) avec persistance et streaming SSE (`/chat/stream`).
- **RAG Hybride (Retrieval-Augmented Generation)** :
  - Base vectorielle **Qdrant** avec embeddings biomédicaux (**S-PubMedBERT-MS-MARCO**).
  - Enrichissement automatique des prompts avec les protocoles neuro-oncologiques et diagnostics différentiels.
- **Sécurité Clinique & Conformité Réglementaire** :
  - **Safety Agent** avec vérification des seuils de confiance (seuil standard : 85%).
  - Audit trail complet et **porte de validation humaine obligatoire** (Approbation / Rejet) conforme à l'**Article 14 de l'EU AI Act**.

---

## 🏗️ Architecture Multi-Agents (LangGraph)

Le cœur applicatif est orchestré par **LangGraph** sous forme de graphe d'états d'agents spécialisés :

```
                        ┌────────────────────────┐
                        │    Controller Agent    │
                        └───────────┬────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Vision Agent     │
                         │  (Attention U-Net /  │
                         │  EfficientNetV2-S)   │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
      ┌──────────────────────┐              ┌──────────────────────┐
      │   Diagnostic Agent   │              │     Safety Agent     │
      │  (Morphométrie, RAG, │              │ (Contrôle qualité &  │
      │   Rapport médical)   │              │   Audit EU AI Act)   │
      └──────────┬───────────┘              └──────────┬───────────┘
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Dialogue Agent    │
                         │ (LLM 4-bit + RAG  │
                         │  + Session Memory)   │
                         └──────────────────────┘
```

1. **Vision Agent** : Prétraite l'image IRM, extrait le masque de segmentation et calcule les probabilités de classe avec carte Grad-CAM.
2. **Diagnostic Agent** : Calcule les métriques morphométriques (surface mm², diamètre max, localisation, ratio tumoral), interroge le RAG et produit la synthèse diagnostique.
3. **Safety Agent** : Évalue la fiabilité, signale les incertitudes (< 85%), déclenche la vérification différentielle et enregistre les logs d'audit.
4. **Dialogue Agent** : Fournit une interface naturelle pour échanger avec le praticien, contextualisée par le rapport d'analyse.
5. **Controller Agent** : Coordonne le flux d'exécution et gère le cycle de vie des sessions.

---

## 🛠️ Stack Technologique

| Couche | Technologie | Rôle |
| :--- | :--- | :--- |
| **Frontend** | Angular 19 | Interface clinique interactive (visualiseur IRM, chat streaming, validation de rapports) |
| **API Gateway** | Spring Boot 3.2 (Java 17) | Sécurité, authentification JWT, passerelle de routage |
| **Backend IA** | FastAPI + Uvicorn | Endpoints REST & SSE, exécution asynchrone des modèles |
| **Orchestration** | LangGraph & LangChain | Graphe d'états et orchestration des agents |
| **Segmentation** | MONAI Attention U-Net / Swin UNETR | Localisation et délimitation de la masse tumorale |
| **Classification** | EfficientNetV2-S | Catégorisation histologique 4 classes |
| **Explainability (XAI)** | Grad-CAM (PyTorch hooks) | Carte thermique d'attention sur les couches convolutives |
| **LLM Dialogue** | LLM Médical Quantifié (4-bit) | Raisonnement médical et échange conversationnel |
| **RAG & Recherche** | Qdrant + S-PubMedBERT | Recherche sémantique vectorielle dans la littérature neuro-oncologique |
| **Base de Données** | MySQL 8.0 + SQLAlchemy | Persistance des analyses, comptes praticiens et historiques de révision |

---

## 📊 Dataset & Notebooks de Recherche

Le projet s'appuie sur le dataset **BRISC 2025** (Brain Tumor MRI Dataset) nettoyé et dédoublonné :
- Nettoyage rigoureux : identification et élimination des doublons par hachage MD5 et filtrage via `clean_manifest.csv` (zéro fuite de données).
- **`notebooks/SwinUNETR.ipynb`** : Entraînement du modèle de segmentation Vision Transformer / Swin UNETR 2D (`segmentation_swinunetr.pth`).
- **`notebooks/classification.ipynb`** : Entraînement du classifieur EfficientNetV2-S (avec explicabilité Grad-CAM, `efficientnet.pth`).
- **`notebooks/dataset_cleaning_analysis.ipynb`** : Audit de qualité, élimination des fuites et génération de `clean_manifest.csv`.

---

## 🚀 Installation & Démarrage

### Prérequis
- Python 3.10+ (ou Python 3.14)
- Node.js 18+ & npm
- Java 17 (pour la passerelle Spring Boot)
- MySQL 8.0+
- GPU NVIDIA (CUDA recommandé, compatible RTX 3050 ~4 Go VRAM)

### 1. Backend Python (FastAPI & Agents)
```bash
# Cloner le dépôt
git clone <repository-url>
cd "projet pcd"

# Activer l'environnement virtuel
# Sur Windows :
.\venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer le fichier d'environnement .env :
# HF_TOKEN=...
# MYSQL_HOST=localhost
# MYSQL_PASSWORD=...

# Lancer le serveur backend
python main.py serve --host 0.0.0.0 --port 8000
```

### 2. Frontend (Angular 19)
```bash
cd frontend
npm install
npm start
# Accessible sur http://localhost:4200
```

### 3. Lancer une analyse directement en ligne de commande (CLI)
```bash
# Analyse d'une image IRM unitaire
python main.py analyze --patient-dir "chemin/vers/image.jpg"
```

---

## 🧪 Tests

Le projet intègre une suite de tests unitaires et d'intégration validant chaque agent et composant :
```bash
# Lancer les tests unitaires de l'agent de dialogue
pytest tests/test_dialogue_agent.py

# Lancer les tests des agents diagnostiques et de sécurité
pytest tests/test_diagnostic_agent.py tests/test_safety_agent.py
```
