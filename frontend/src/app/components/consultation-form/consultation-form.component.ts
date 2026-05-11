import { Component, EventEmitter, Input, Output } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, X, Save, ClipboardList } from 'lucide-angular';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-consultation-form',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './consultation-form.component.html'
})
export class ConsultationFormComponent {
  readonly XIcon = X;
  readonly SaveIcon = Save;
  readonly ClipboardListIcon = ClipboardList;

  @Input() patients: any[] = [];
  @Input() medecins: any[] = [];
  @Output() close = new EventEmitter<void>();
  @Output() saved = new EventEmitter<void>();

  formData = {
    patient_id: '',
    medecin_id: '',
    motif: '',
    diagnostic: '',
    notes: '',
    rapport_genere: '',
    statut: 'en_cours'
  };
  loading = false;
  error: string | null = null;
  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  handleSubmit(e: Event) {
    e.preventDefault();
    if (!this.formData.patient_id || !this.formData.medecin_id) {
      this.error = "Le patient et le médecin sont obligatoires.";
      return;
    }

    this.loading = true;
    this.error = null;

    const payload = {
      patient_id: parseInt(this.formData.patient_id),
      medecin_id: parseInt(this.formData.medecin_id),
      motif: this.formData.motif || null,
      diagnostic: this.formData.diagnostic || null,
      notes: this.formData.notes || null,
      rapport_genere: this.formData.rapport_genere || null,
      statut: this.formData.statut
    };

    this.http.post(`${this.API_BASE}/consultations`, payload).subscribe({
      next: () => {
        this.saved.emit();
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Échec de la création de la consultation.';
        this.loading = false;
      }
    });
  }
}
