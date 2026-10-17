# 🧠 NeuroScan AI — Plateforme d'Analyse d'IRM Cérébrales par IA Multi-Agents

**Système Hospitalier Intelligent pour la Détection, Segmentation 2D, Classification Multiclasse, Explicabilité (Grad-CAM) et Dialogue Clinique.**

Développé et optimisé sur le dataset **BRISC 2025** (coupes axiales IRM pondérées T1).

---

## 📋 Aperçu du Projet

**NeuroScan AI** est une station de travail clinique avancée conçue pour assister les neuro-radiologues, neurologues et neurochirurgiens. Elle combine des modèles de vision par ordinateur de pointe, une architecture multi-agents orchestrée par **LangGraph**, un assistant conversationnel médical local, une base de connaissances vectorielle **RAG (Qdrant)**, et une passerelle sécurisée **Spring Boot + JWT** connectée à un frontend moderne **Angular 19**.

L'application respecte les exigences de traçabilité et de supervision humaine imposées par l'**Article 14 de l'EU AI Act** (Human-in-the-Loop).

---

## 🌟 Fonctionnalités Principales

### 1. Imagerie Médicale & Vision par Ordinateur (2D PACS)
- **Segmentation Tumorale 2D** :
  - Architecture **U-Net** avec backbone Transformer **SegFormer (MiT-B2)** / Swin UNETR (`segmentation_swinunetr.pth`).
  - Délimitation précise des contours tumoraux et calcul morphométrique automatique (surface en mm², diamètre maximal en mm, ratio tumoral, intensités de signal T1, quadrant anatomique).
- **Classification Histologique 4 Classes** :
  - Architecture **EfficientNetV2-S** (`efficientnet.pth`) atteignant ~98% de précision.
  - Diagnostic différentiel : **Gliome**, **Méningiome**, **Tumeur pituitaire (adénome hypophysaire)**, ou **Absence de tumeur (sain)**.
- **Explicabilité Visuelle (Grad-CAM XAI)** :
  - Génération de cartes d'attention thermique sur les dernières couches convolutionnelles pour visualiser les régions critiques ayant motivé la classification de l'IA.
- **Console PACS 2D Interactive** :
  - Basculement instantané entre les modes : **IRM Brute**, **Overlay Segmentation 2D** (fusion haute netteté de la coupe axiale et du masque tumoral), et **Grad-CAM XAI**.
  - Zone de glisser-déposer fluide pour les coupes axiales (JPG/PNG).
  - Sélection directe du patient depuis la base de données hospitalière.

### 2. Synthèse Clinique & Rapports Médicaux
- **Génération Déterministe et Réglementaire** :
  - Rapport structuré en sections médicales normalisées : *Indication*, *Technique*, *Résultats*, *Interprétation*, *Conclusion*, et *Recommandations*.
  - **Découplé du RAG** : les comptes rendus d'examen reposent strictement sur les constatations morphologiques objectives de l'image et les règles de gradation OMS pour garantir l'absence totale d'hallucinations documentaires.
- **Export PDF Officiel** :
  - Génération et impression en un clic du rapport médical complet intégrant les métadonnées patient, les mesures morphométriques, la synthèse rédigée et la capture IRM avec overlay IA.

### 3. Assistant Conversationnel Clinique (Dialogue Agent)
- Modèle LLM médical local servi via **Ollama** (`hermes3:3b`, sans envoi de données de santé vers des API tierces).
- Gestion multi-conversations isolées avec mémoire de session.
- Contextualisation automatique avec les résultats de l'IRM du patient actif.

### 4. Base de Connaissances RAG Hybride (Qdrant)
- Base de données vectorielle **Qdrant** persistante (`data/qdrant_storage`).
- Modèle d'embeddings biomédical spécialisé : **`pritamdeka/S-PubMedBert-MS-MARCO`** (768 dimensions).
- Module d'ingestion documentaire avec support **OCR EasyOCR** (fichiers PDF, DOCX, TXT, images) pour indexer les protocoles hospitaliers et publications scientifiques.
- Utilisé par l'Agent de Dialogue pour répondre aux questions cliniques spécialisées et citer les sources.

### 5. Sécurité Clinique & Piste d'Audit (EU AI Act)
- **Safety Agent** : Détection des incertitudes diagnostiques (confiance < 85%) et alertes différentielles.
- **Porte de Contrôle Humain (Gatekeeper Art. 14)** :
  - Obligation de relecture par le médecin : **Valider** le rapport ou déclencher une **Seconde Lecture** avec motif clinique consigné dans la base d'audit.

---

## 🏗️ Architecture Multi-Agents (LangGraph)

Le cycle d'inférence diagnostique est orchestré sous la forme d'un graphe d'états :

```
                          ┌────────────────────────┐
                          │    Controller Agent    │
                          └───────────┬────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │     Vision Agent     │
                           │  - Segmentation 2D   │
                           │  - EfficientNetV2-S  │
                           │  - Grad-CAM Heatmap  │
                           └──────────┬───────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌──────────────────────┐              ┌──────────────────────┐
        │   Diagnostic Agent   │              │     Safety Agent     │
        │ - Morphométrie 2D    │              │ - Seuils de confiance│
        │ - Gradation OMS      │              │ - Audit EU AI Act    │
        │ - Synthèse clinique  │              │ - Validation humaine │
        └──────────┬───────────┘              └──────────┬───────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Dialogue Agent    │
                           │ (LLM Ollama + RAG    │
                           │  Vectoriel Qdrant)   │
                           └──────────────────────┘
```

---

## 🖥️ Modules de l'Application

L'interface web Angular 19 propose un design épuré, lumineux (*Light Theme*) et hospitalier :

| Module | Route | Description |
| :--- | :--- | :--- |
| **Gestion Patients** | `/` | Dossiers médicaux, liste des patients, antécédents et consultations liées |
| **Gestion Médecins** | `/medecins` | Annuaire des praticiens, spécialités (neuroradiologie, neurochirurgie) |
| **Gestion Consultations** | `/consultations` | Prise de rendez-vous, suivi clinique et observations |
| **Gestion Examens** | `/analysis` | Console PACS 2D, analyse en temps réel, explicabilité et export PDF |
| **Dialogue Agent** | `/chat` | Assistant IA interactif connecté aux antécédents du patient |
| **RAG Knowledge Base**| `/knowledge` | Import de documents cliniques, OCR et indexation vectorielle |
| **Authentification** | `/login`, `/register` | Inscription et connexion sécurisées pour les professionnels de santé |

---

## 🛠️ Stack Technologique

- **Frontend** : Angular 19, TypeScript, Lucide Icons, CSS3 Glassmorphism (Thème clair clinique).
- **Passerelle d'API** : Spring Boot 3.2.4 (Java 17), Spring Security, JJWT, Apache HttpClient.
- **Backend IA** : FastAPI, Uvicorn, Python 3.10 / 3.14, Pydantic v2.
- **Deep Learning** : PyTorch, Torchvision, Segmentation Models PyTorch (SMP), Timm.
- **Orchestration IA** : LangGraph, LangChain.
- **LLM Local** : Ollama (`hermes3:3b`).
- **Embeddings & Vector Store** : Sentence-Transformers (`S-PubMedBert-MS-MARCO`), Qdrant Client.
- **Base de Données** : MySQL 8.0, SQLAlchemy, Hibernate / JPA.

---

## 📊 Dataset & Notebooks de Recherche

Le projet s'appuie sur le dataset **BRISC 2025** (Brain Tumor MRI Dataset) nettoyé et dédoublonné :
- **`notebooks/dataset_cleaning_analysis.ipynb`** : Détection et élimination des doublons par hachage MD5, élimination des fuites inter-partitions et génération du manifeste propre `clean_manifest.csv`.
- **`notebooks/SwinUNETR.ipynb`** : Recherche et entraînement des architectures de segmentation 2D (`segmentation_swinunetr.pth`).
- **`notebooks/classification.ipynb`** : Entraînement du classifieur EfficientNetV2-S avec évaluation multiclasse et calcul Grad-CAM (`efficientnet.pth`).
- **`notebooks/analyse-segmentation.ipynb`** : Évaluation quantitative des prédictions (Dice score, IoU).

---

## 🚀 Installation & Démarrage

### Prérequis
- Python 3.10+ (ou Python 3.14)
- Node.js 18+ & npm
- Java 17 & Maven
- MySQL 8.0+
- Ollama (avec le modèle `ollama pull hermes3:3b`)
- GPU NVIDIA (CUDA recommandé)

---

### 1. Base de Données MySQL
Créez la base de données MySQL :
```sql
CREATE DATABASE IF NOT EXISTS pcd_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

---

### 2. Backend IA (FastAPI)
```bash
# Cloner le dépôt
git clone https://github.com/Rabie62/projet_pcd.git
cd "projet pcd"

# Activer l'environnement virtuel (Windows)
.\venv\Scripts\activate

# Installer les dépendances Python
pip install -r requirements.txt

# Configurer les variables d'environnement (.env)
cp .env.example .env

# Lancer le serveur backend IA (port 8000 par défaut)
python main.py
# ou avec options :
python main.py serve --host 0.0.0.0 --port 8000
```

---

### 3. Passerelle Backend (Spring Boot)
```bash
cd spring-backend
mvn clean spring-boot:run
# Le serveur démarre sur http://localhost:8080
```

---

### 4. Frontend (Angular 19)
```bash
cd frontend
npm install
npm start
# L'application est accessible sur http://localhost:4200
```

---

### 5. CLI — Analyse d'IRM en ligne de commande
Vous pouvez analyser directement une image IRM via le terminal sans passer par l'interface web :
```bash
python main.py analyze --patient-dir "chemin/vers/coupe_irm.jpg" -o "outputs/resultat.png"
```

---

## 🧪 Tests Unitaires et d'Intégration

Le projet inclut une couverture de tests automatisés validant l'ensemble de la chaîne :
```bash
# Exécuter l'intégralité des tests
pytest tests/

# Tester spécifiquement l'Agent Diagnostique
pytest tests/test_diagnostic_agent.py

# Tester les flux d'intégration de l'API
pytest tests/test_integration.py

# Tester la sécurité et l'EU AI Act Gatekeeper
pytest tests/test_safety_agent.py
```

---

## ⚖️ Conformité et Clause Réglementaire

> **Avertissement Médical & Réglementaire (EU AI Act)** :  
> Ce logiciel est un dispositif d'aide à la décision clinique assisté par intelligence artificielle. Les segmentations, probabilités de classification et propositions de synthèses générées ne constituent en aucun cas un diagnostic médical définitif. Toute conclusion doit impérativement faire l'objet d'une validation par un médecin spécialiste (radiologue / neurochirurgien) avant toute décision thérapeutique ou intervention chirurgicale.
