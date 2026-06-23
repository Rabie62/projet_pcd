import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Search, Plus, Trash2 } from 'lucide-angular';
import { Subject, takeUntil } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ErrorHandlerService } from '../../services/error-handler.service';
import { Patient } from '../../models';
import { PatientRegistryFormComponent } from '../patient-registry-form/patient-registry-form.component';

@Component({
  selector: 'app-patient-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, PatientRegistryFormComponent],
  templateUrl: './patient-dashboard.component.html',
  styleUrls: ['./patient-dashboard.component.css']
})
export class PatientDashboardComponent implements OnInit, OnDestroy {
  readonly SearchIcon = Search;
  readonly PlusIcon = Plus;
  readonly Trash2Icon = Trash2;

  patients: Patient[] = [];
  loading = true;
  showRegistry = false;
  selectedIds = new Set<number>();
  searchQuery = '';

  private destroy$ = new Subject<void>();
  private API_BASE = environment.apiUrl;

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService
  ) {}

  ngOnInit(): void {
    this.fetchPatients();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  fetchPatients(): void {
    this.loading = true;
    this.http.get<any>(`${this.API_BASE}/patients`, { withCredentials: true })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (data) => {
          this.patients = data.items ?? data ?? [];
          this.loading = false;
        },
        error: (err) => {
          this.errorHandler.handleError(err);
          this.loading = false;
        }
      });
  }

  get filteredPatients(): Patient[] {
    if (!this.searchQuery.trim()) return this.patients;
    const query = this.searchQuery.toLowerCase();
    return this.patients.filter(p =>
      p.nom.toLowerCase().includes(query) ||
      p.prenom.toLowerCase().includes(query) ||
      p.id.toString().includes(query)
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

  handleRemoveSelected(): void {
    if (this.selectedIds.size === 0) return;
    const confirmMessage = `Êtes-vous sûr de vouloir supprimer ${this.selectedIds.size} patient(s) ?\n\nCette action supprimera définitivement tous les historiques d'examens et rapports associés. Cette action est irréversible.`;

    if (!window.confirm(confirmMessage)) return;

    this.loading = true;
    const ids = Array.from(this.selectedIds);

    for (const id of ids) {
      this.http.delete(`${this.API_BASE}/patients/${id}`, { withCredentials: true })
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
              this.fetchPatients();
            }
          }
        });
    }
  }
}
