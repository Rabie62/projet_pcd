import { Component, EventEmitter, Input, Output, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, X, Save, UserPlus } from 'lucide-angular';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-medecin-form',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './medecin-form.component.html'
})
export class MedecinFormComponent implements OnInit {
  readonly XIcon = X;
  readonly SaveIcon = Save;
  readonly UserPlusIcon = UserPlus;

  @Input() medecin: any = null;
  @Output() close = new EventEmitter<void>();
  @Output() saved = new EventEmitter<void>();

  isEdit = false;
  formData = {
    nom: '',
    prenom: '',
    specialite: '',
    tel: '',
    email: '',
    departement: ''
  };
  loading = false;
  error: string | null = null;
  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    if (this.medecin) {
      this.isEdit = true;
      this.formData = {
        nom: this.medecin.nom || '',
        prenom: this.medecin.prenom || '',
        specialite: this.medecin.specialite || '',
        tel: this.medecin.tel || '',
        email: this.medecin.email || '',
        departement: this.medecin.departement || ''
      };
    }
  }

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
      specialite: this.formData.specialite || null,
      tel: this.formData.tel || null,
      email: this.formData.email || null,
      departement: this.formData.departement || null
    };

    const request = this.isEdit
      ? this.http.put(`${this.API_BASE}/medecins/${this.medecin.id}`, payload)
      : this.http.post(`${this.API_BASE}/medecins`, payload);

    request.subscribe({
      next: () => {
        this.saved.emit();
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Échec de l\'enregistrement.';
        this.loading = false;
      }
    });
  }
}
