import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Search, Plus, Trash2, Edit3, X, Save } from 'lucide-angular';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';
import { MedecinFormComponent } from '../medecin-form/medecin-form.component';

@Component({
  selector: 'app-medecin-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, MedecinFormComponent],
  templateUrl: './medecin-dashboard.component.html',
  styleUrls: ['../patient-dashboard/patient-dashboard.component.css']
})
export class MedecinDashboardComponent implements OnInit {
  readonly SearchIcon = Search;
  readonly PlusIcon = Plus;
  readonly Trash2Icon = Trash2;
  readonly Edit3Icon = Edit3;

  medecins: any[] = [];
  loading = true;
  showForm = false;
  editingMedecin: any = null;
  selectedIds = new Set<number>();
  searchQuery = '';

  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchMedecins();
  }

  fetchMedecins() {
    this.loading = true;
    this.http.get<any[]>(`${this.API_BASE}/medecins`).subscribe({
      next: (data) => {
        this.medecins = data;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }

  get filteredMedecins(): any[] {
    if (!this.searchQuery.trim()) return this.medecins;
    const query = this.searchQuery.toLowerCase();
    return this.medecins.filter(m =>
      m.nom.toLowerCase().includes(query) ||
      m.prenom.toLowerCase().includes(query) ||
      m.specialite?.toLowerCase().includes(query) ||
      m.departement?.toLowerCase().includes(query) ||
      m.id.toString().includes(query)
    );
  }

  toggleSelect(id: number, event?: Event) {
    if (event) event.stopPropagation();
    if (this.selectedIds.has(id)) {
      this.selectedIds.delete(id);
    } else {
      this.selectedIds.add(id);
    }
  }

  toggleSelectAll() {
    const filtered = this.filteredMedecins;
    if (this.isSelectedAll()) {
      filtered.forEach(m => this.selectedIds.delete(m.id));
    } else {
      filtered.forEach(m => this.selectedIds.add(m.id));
    }
  }

  isSelectedAll(): boolean {
    const filtered = this.filteredMedecins;
    return filtered.length > 0 && filtered.every(m => this.selectedIds.has(m.id));
  }

  openAddForm() {
    this.editingMedecin = null;
    this.showForm = true;
  }

  openEditForm(medecin: any, event?: Event) {
    if (event) event.stopPropagation();
    this.editingMedecin = medecin;
    this.showForm = true;
  }

  handleFormClose() {
    this.showForm = false;
    this.editingMedecin = null;
  }

  handleFormSaved() {
    this.fetchMedecins();
    this.showForm = false;
    this.editingMedecin = null;
  }

  async handleRemoveSelected() {
    if (this.selectedIds.size === 0) return;
    const confirmMessage = `Êtes-vous sûr de vouloir supprimer ${this.selectedIds.size} médecin(s) ?\n\nToutes les consultations associées seront également supprimées. Cette action est irréversible.`;
    if (!window.confirm(confirmMessage)) return;

    this.loading = true;
    const ids = Array.from(this.selectedIds);
    try {
      for (const id of ids) {
        await firstValueFrom(this.http.delete(`${this.API_BASE}/medecins/${id}`));
      }
      this.selectedIds.clear();
      this.fetchMedecins();
    } catch (err) {
      console.error('Failed to delete doctors', err);
      alert('Certains médecins n\'ont pas pu être supprimés.');
      this.loading = false;
    }
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
      default: return statut;
    }
  }
}
