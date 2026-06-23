// ── Patient types ──
export interface Patient {
  id: number;
  prenom: string;
  nom: string;
  date_naissance?: string | null;
  genre?: string | null;
  tel?: string | null;
  poids?: number | null;
  taille?: number | null;
  FC?: number | null;
  glycemie?: number | null;
  scan_count: number;
  age?: number | null;
  created_at: string;
  updated_at?: string | null;
}

export interface PatientCreateRequest {
  prenom: string;
  nom: string;
  date_naissance?: string;
  genre?: string;
  tel?: string;
  poids?: number;
  taille?: number;
  FC?: number;
  glycemie?: number;
}

// ── Doctor types ──
export interface Medecin {
  id: number;
  nom: string;
  prenom: string;
  specialite: string | null;
  tel: string | null;
  email: string | null;
  departement: string | null;
  username: string;
  consultation_count: number;
  created_at: string;
  updated_at: string | null;
}

// ── Consultation types ──
export interface Consultation {
  id: number;
  patient_id: number;
  patient_nom: string | null;
  patient_prenom: string | null;
  medecin_id: number;
  medecin_nom: string | null;
  medecin_prenom: string | null;
  medecin_specialite: string | null;
  scan_record_id: number | null;
  session_id: string | null;
  date_consultation: string;
  motif: string | null;
  diagnostic: string | null;
  notes: string | null;
  rapport_genere: string | null;
  statut: string | null;
  created_at: string;
  updated_at: string | null;
}

// ── Scan Analysis types ──
export interface TumorFeatures {
  tumor_area_mm2: number;
  max_diameter_mm: number;
  tumor_ratio: number;
  location_description: string;
}

export interface SafetyCheck {
  passed: boolean;
  confidence_adequate: boolean;
  requires_human_review: boolean;
  flags: string[];
  warnings: string[];
  compliance_notes: string[];
}

export interface AnalysisResponse {
  session_id: string;
  patient_id: string;
  status: string;
  tumor_detected: boolean;
  classification: string | null;
  classification_confidence: number;
  who_grade: string | null;
  tumor_features: TumorFeatures | null;
  clinical_summary: string;
  recommendations: string[];
  safety_check: SafetyCheck | null;
  review_status: string | null;
  disclaimer: string;
  errors: string[];
}

// ── Knowledge Base types ──
export interface KnowledgeDocument {
  document_id: string;
  filename: string;
  uploaded_by: string;
  uploaded_at: string;
  file_type: string;
  chunk_count: number;
}

export interface KnowledgeStatus {
  available: boolean;
  total_chunks: number;
  uploaded_documents: number;
  system_knowledge_indexed: boolean;
}

// ── Auth types ──
export interface AuthUser {
  id: number;
  nom: string;
  prenom: string;
  email: string | null;
  specialite: string | null;
  departement: string | null;
  username: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  nom: string;
  prenom: string;
  specialite?: string;
  tel?: string;
  email?: string;
  departement?: string;
  username: string;
  password: string;
}
