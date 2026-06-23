import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Search, Plus, Trash2, Edit3 } from 'lucide-angular';
import { Subject, takeUntil } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ErrorHandlerService } from '../../services/error-handler.service';
import { Medecin } from '../../models';
import { MedecinFormComponent } from '../medecin-form/medecin-form.component';

@Component({
  selector: 'app-medecin-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, MedecinFormComponent],
  templateUrl: './medecin-dashboard.component.html',
  styleUrls: ['../patient-dashboard/patient-dashboard.component.css']
})
export class MedecinDashboardComponent implements OnInit, OnDestroy {
  readonly SearchIcon = Search;
  readonly PlusIcon = Plus;
  readonly Trash2Icon = Trash2;
  readonly Edit3Icon = Edit3;

  medecins: Medecin[] = [];
  loading = true;
  showForm = false;
  editingMedecin: Medecin | null = null;
  selectedIds = new Set<number>();
  searchQuery = '';

  private destroy$ = new Subject<void>();
  private API_BASE = environment.apiUrl;

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService
  ) {}

  ngOnInit(): void {
    this.fetchMedecins();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  fetchMedecins(): void {
    this.loading = true;
    this.http.get<any>(`${this.API_BASE}/medecins`, { withCredentials: true })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data) => {
          this.medecins = data.items ?? data ?? [];
          this.loading = false;
        },
        error: (err) => {
          this.errorHandler.handleError(err);
          this.loading = false;
        }
      });
  }

  get filteredMedecins(): Medecin[] {
    if (!this.searchQuery.trim()) return this.medecins;
    const query = this.searchQuery.toLowerCase();
    return this.medecins.filter(m =>
      m.nom.toLowerCase().includes(query) ||
      m.prenom.toLowerCase().includes(query) ||
      (m.specialite || '').toLowerCase().includes(query) ||
      (m.departement || '').toLowerCase().includes(query) ||
      m.id.toString().includes(query)
    );
  }

  toggleSelect(id: number, event?: Event): void {
    if (event) event.stopPropagation();
    if (this.selectedIds.has(id)) {
      this.selectedIds.delete(id);
    } else {
      this.selectedIds.add(id);
    }
  }

  toggleSelectAll(): void {
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

  openAddForm(): void {
    this.editingMedecin = null;
    this.showForm = true;
  }

  openEditForm(medecin: Medecin, event?: Event): void {
    if (event) event.stopPropagation();
    this.editingMedecin = medecin;
    this.showForm = true;
  }

  handleFormClose(): void {
    this.showForm = false;
    this.editingMedecin = null;
  }

  handleFormSaved(): void {
    this.fetchMedecins();
    this.showForm = false;
    this.editingMedecin = null;
  }

  handleRemoveSelected(): void {
    if (this.selectedIds.size === 0) return;
    const confirmMessage = `Êtes-vous sûr de vouloir supprimer ${this.selectedIds.size} médecin(s) ?\n\nToutes les consultations associées seront également supprimées. Cette action est irréversible.`;

    if (!window.confirm(confirmMessage)) return;

    this.loading = true;
    const ids = Array.from(this.selectedIds);

    for (const id of ids) {
      this.http.delete(`${this.API_BASE}/medecins/${id}`, { withCredentials: true })
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: () => {
            this.selectedIds.delete(id);
          },
          error: (err) => {
            this.errorHandler.handleError(err);
            this.loading = false;
          },
          complete: () => {
            if (this.selectedIds.size === 0) {
              this.fetchMedecins();
            }
          }
        });
    }
  }
}
