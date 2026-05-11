# 🧠 Medical AI Agent — Analyse de Tumeurs Cérébrales par IRM

**Système Multi-Agent d’Intelligence Artificielle pour la Détection, Segmentation, Classification et Génération de Rapports Cliniques.**

Développé sur le dataset **BRISC 2025** (images IRM 2D pondérées T1).

---

## 📋 Aperçu

**Medical AI Agent** est une plateforme médicale avancée qui automatise l’analyse des IRM cérébrales. Il combine vision par ordinateur, deep learning et agents intelligents pour assister les radiologues dans le diagnostic des tumeurs cérébrales.

### Fonctionnalités principales

- **Segmentation** précise de la tumeur à l’aide d’**Attention U-Net** (MONAI)
- **Classification** multiclasse : Gliome, Méningiome, Pituitaire, Pas de tumeur (ResNet-18)
- **Génération automatique** de rapports radiologiques structurés en français (Gemma-3 1B-IT)
- **RAG** (Retrieval-Augmented Generation) avec connaissances cliniques vérifiées
- **Agent de Sécurité** avec seuils de confiance et audit complet
- **Interface conversationnelle** avec l’IA médicale
- **Workflow avec validation humaine obligatoire** (conforme EU AI Act)

---

## 🏗️ Architecture

Le système repose sur une **architecture multi-agents orchestrée par LangGraph** :

- **Vision Agent** → Segmentation + Classification
- **Diagnostic Agent** → Analyse morphométrique, RAG et génération de rapport
- **Safety Agent** → Contrôles qualité, détection d’anomalies et audit
- **Dialogue Agent** → Chat contextuel avec le médecin
- **Controller Agent** → Orchestration et gestion des sessions

**Frontend** : Angular 19  
**Backend** : FastAPI (Python) + Spring Boot Gateway (Java 17)

---

## 🛠️ Stack Technologique

| Couche                | Technologie                              |
|-----------------------|------------------------------------------|
| **Frontend**          | Angular 19                               |
| **API Gateway**       | Spring Boot 3.2 (Java 17)                |
| **Backend**           | FastAPI + Uvicorn                        |
| **Orchestration**     | LangGraph + LangChain                    |
| **Segmentation**      | MONAI Attention U-Net                    |
| **Classification**    | ResNet-18 (Torchvision)                  |
| **LLM**               | Google Gemma-3 1B-IT                     |
| **RAG**               | Qdrant + S-PubMedBERT-MS-MARCO           |
| **Base de données**   | MySQL + SQLAlchemy                       |
| **Codage médical**    | WHO ICD-11 API                           |

---

## 🚀 Installation & Démarrage

### Prérequis

- Python 3.10+
- Node.js 18+ & npm
- Java 17
- MySQL 8.0+
- GPU CUDA (recommandé)

### Étapes d'installation

```bash
# 1. Cloner le repository
git clone <repository-url>
cd projet-pcd

# 2. Environnement virtuel Python
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt

# 3. Frontend
cd frontend
npm install
npm start                         # http://localhost:4200

# 4. Backend (dans un autre terminal)
cd ..                             
python main.py serve              # http://localhost:8000
