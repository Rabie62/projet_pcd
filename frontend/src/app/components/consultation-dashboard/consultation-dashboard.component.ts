import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Search, Plus, Trash2, Eye, X, FileText, Calendar, User } from 'lucide-angular';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';
import { ConsultationFormComponent } from '../consultation-form/consultation-form.component';
import { ConsultationDetailComponent } from '../consultation-detail/consultation-detail.component';

@Component({
  selector: 'app-consultation-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, ConsultationFormComponent, ConsultationDetailComponent],
  templateUrl: './consultation-dashboard.component.html',
  styleUrls: ['../patient-dashboard/patient-dashboard.component.css']
})
export class ConsultationDashboardComponent implements OnInit {
  readonly SearchIcon = Search;
  readonly PlusIcon = Plus;
  readonly Trash2Icon = Trash2;
  readonly EyeIcon = Eye;
  readonly FileTextIcon = FileText;
  readonly CalendarIcon = Calendar;
  readonly UserIcon = User;

  consultations: any[] = [];
  medecins: any[] = [];
  patients: any[] = [];
  loading = true;
  showForm = false;
  selectedConsultation: any = null;
  selectedIds = new Set<number>();
  searchQuery = '';
  filterPatient = '';
  filterStatut = '';
  selectedPatientId: number | null = null;

  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchConsultations();
    this.fetchMedecins();
    this.fetchPatients();
  }

  fetchConsultations() {
    this.loading = true;
    const params: any = {};
    if (this.selectedPatientId) params.patient_id = this.selectedPatientId;
    this.http.get<any[]>(`${this.API_BASE}/consultations`, { params }).subscribe({
      next: (data) => {
        this.consultations = data;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }

  fetchMedecins() {
    this.http.get<any[]>(`${this.API_BASE}/medecins`).subscribe({
      next: (data) => { this.medecins = data; },
      error: (err) => { console.error(err); }
    });
  }

  fetchPatients() {
    this.http.get<any[]>(`${this.API_BASE}/patients`).subscribe({
      next: (data) => { this.patients = data; },
      error: (err) => { console.error(err); }
    });
  }

  get filteredConsultations(): any[] {
    let results = this.consultations;

    if (this.searchQuery.trim()) {
      const query = this.searchQuery.toLowerCase();
      results = results.filter(c =>
        (c.patient_nom || '').toLowerCase().includes(query) ||
        (c.patient_prenom || '').toLowerCase().includes(query) ||
        (c.medecin_nom || '').toLowerCase().includes(query) ||
        (c.medecin_prenom || '').toLowerCase().includes(query) ||
        (c.motif || '').toLowerCase().includes(query) ||
        c.id.toString().includes(query)
      );
    }

    if (this.filterStatut) {
      results = results.filter(c => c.statut === this.filterStatut);
    }

    return results;
  }

  toggleSelect(id: number, event?: Event) {
    if (event) event.stopPropagation();
    const newSet = new Set(this.selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    this.selectedIds = newSet;
  }

  toggleSelectAll() {
    const filtered = this.filteredConsultations;
    const newSet = new Set<number>();
    if (!this.isSelectedAll()) {
      filtered.forEach(c => newSet.add(c.id));
    }
    this.selectedIds = newSet;
  }

  isSelectedAll(): boolean {
    const filtered = this.filteredConsultations;
    return filtered.length > 0 && filtered.every(c => this.selectedIds.has(c.id));
  }

  viewDetail(consultation: any) {
    this.selectedConsultation = consultation;
  }

  closeDetail() {
    this.selectedConsultation = null;
  }

  openAddForm() {
    this.showForm = true;
  }

  handleFormSaved() {
    this.fetchConsultations();
    this.showForm = false;
  }

  handleFormClose() {
    this.showForm = false;
  }

  async handleRemoveSelected() {
    if (this.selectedIds.size === 0) return;
    if (!window.confirm(`Supprimer ${this.selectedIds.size} consultation(s) ? Cette action est irréversible.`)) return;

    this.loading = true;
    const ids = Array.from(this.selectedIds);
    try {
      for (const id of ids) {
        await firstValueFrom(this.http.delete(`${this.API_BASE}/consultations/${id}`));
      }
      this.selectedIds = new Set();
      this.fetchConsultations();
    } catch (err) {
      console.error('Failed to delete consultations', err);
      alert('Certaines consultations n\'ont pas pu être supprimées.');
      this.loading = false;
    }
  }

  onPatientFilterChange() {
    this.selectedPatientId = this.filterPatient ? parseInt(this.filterPatient) : null;
    this.fetchConsultations();
  }

  getStatutClass(statut: string): string {
    switch (statut) {
      case 'terminee': return 'safe';
      case 'en_cours': return 'warning';
      case 'annulee': return 'danger';
      default: return '';
    }
  }

  getStatutLabel(statut: string): string {
    switch (statut) {
      case 'terminee': return 'Terminée';
      case 'en_cours': return 'En cours';
      case 'annulee': return 'Annulée';
      default: return statut || '—';
    }
  }

  formatDate(dateStr: string): string {
    if (!dateStr) return '—';
    const date = new Date(dateStr);
    return date.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' });
  }

  formatDateTime(dateStr: string): string {
    if (!dateStr) return '—';
    const date = new Date(dateStr);
    return date.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
}
