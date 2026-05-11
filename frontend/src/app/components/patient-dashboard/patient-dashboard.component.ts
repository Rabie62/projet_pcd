import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Search, Plus, Trash2 } from 'lucide-angular';
import { environment } from '../../../environments/environment';
import { firstValueFrom } from 'rxjs';
import { PatientRegistryFormComponent } from '../patient-registry-form/patient-registry-form.component';

@Component({
  selector: 'app-patient-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, PatientRegistryFormComponent],
  templateUrl: './patient-dashboard.component.html'
})
export class PatientDashboardComponent implements OnInit {
  readonly SearchIcon = Search;
  readonly PlusIcon = Plus;
  readonly Trash2Icon = Trash2;

  patients: any[] = [];
  loading = true;
  showRegistry = false;
  selectedIds = new Set<number>();
  searchQuery = '';

  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchPatients();
  }

  fetchPatients() {
    this.loading = true;
    this.http.get<any[]>(`${this.API_BASE}/patients`).subscribe({
      next: (data) => {
        this.patients = data;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }

  get filteredPatients(): any[] {
    if (!this.searchQuery.trim()) {
      return this.patients;
    }
    const query = this.searchQuery.toLowerCase();
    return this.patients.filter(p => 
      p.nom.toLowerCase().includes(query) || 
      p.prenom.toLowerCase().includes(query) ||
      p.id.toString().includes(query)
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
    const filtered = this.filteredPatients;
    if (this.isSelectedAll()) {
      filtered.forEach(p => this.selectedIds.delete(p.id));
    } else {
      filtered.forEach(p => this.selectedIds.add(p.id));
    }
  }

  isSelectedAll(): boolean {
    const filtered = this.filteredPatients;
    return filtered.length > 0 && filtered.every(p => this.selectedIds.has(p.id));
  }

  async handleRemoveSelected() {
    if (this.selectedIds.size === 0) return;
    const confirmMessage = `Êtes-vous sûr de vouloir supprimer ${this.selectedIds.size} patient(s) ? \n\nCette action supprimera définitivement tous les historiques d'examens et rapports associés. Cette action est irréversible.`;
    
    if (!window.confirm(confirmMessage)) return;

    this.loading = true;
    
    const ids = Array.from(this.selectedIds);
    try {
      for (const id of ids) {
         await firstValueFrom(this.http.delete(`${this.API_BASE}/patients/${id}`));
      }
      this.selectedIds.clear();
      this.fetchPatients();
    } catch (err) {
      console.error("Failed to delete patients", err);
      alert("Certains dossiers patients n'ont pas pu être supprimés. Veuillez vérifier les logs.");
      this.loading = false;
    }
  }


}
