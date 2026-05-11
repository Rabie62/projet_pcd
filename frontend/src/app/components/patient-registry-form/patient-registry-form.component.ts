import { Component, EventEmitter, Output, Input } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, X, Save, User } from 'lucide-angular';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-patient-registry-form',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './patient-registry-form.component.html'
})
export class PatientRegistryFormComponent {
  readonly XIcon = X;
  readonly SaveIcon = Save;
  readonly UserIcon = User;

  @Output() closeForm = new EventEmitter<void>();
  @Output() refreshList = new EventEmitter<void>();

  formData = {
    prenom: '',
    nom: '',
    date_naissance: '',
    genre: 'M',
    tel: '',
    poids: '',
    taille: '',
    FC: '',
    glycemie: ''
  };

  loading = false;
  error: string | null = null;
  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) { }

  handleSubmit(e: Event) {
    e.preventDefault();
    if (!this.formData.nom || !this.formData.prenom) {
      this.error = "Le prénom et le nom sont obligatoires.";
      return;
    }

    this.loading = true;
    this.error = null;

    const payload = {
      nom: this.formData.nom,
      prenom: this.formData.prenom,
      date_naissance: this.formData.date_naissance || null,
      genre: this.formData.genre,
      tel: this.formData.tel || null,
      poids: this.formData.poids ? parseFloat(this.formData.poids) : null,
      taille: this.formData.taille ? parseFloat(this.formData.taille) : null,
      FC: this.formData.FC ? parseInt(this.formData.FC) : null,
      glycemie: this.formData.glycemie ? parseFloat(this.formData.glycemie) : null
    };

    // The Spring Boot backend forwards /patients
    this.http.post(`${this.API_BASE}/patients`, payload).subscribe({
      next: () => {
        this.refreshList.emit();
        this.closeForm.emit();
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || "Failed to register patient.";
        this.loading = false;
      }
    });
  }
}
