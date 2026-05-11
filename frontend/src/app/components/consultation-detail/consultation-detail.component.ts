import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { LucideAngularModule, X, FileText, Brain, Calendar, User, Activity, ClipboardList } from 'lucide-angular';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-consultation-detail',
  standalone: true,
  imports: [CommonModule, LucideAngularModule],
  templateUrl: './consultation-detail.component.html'
})
export class ConsultationDetailComponent implements OnInit, OnChanges {
  readonly XIcon = X;
  readonly FileTextIcon = FileText;
  readonly BrainIcon = Brain;
  readonly CalendarIcon = Calendar;
  readonly UserIcon = User;
  readonly ActivityIcon = Activity;
  readonly ClipboardListIcon = ClipboardList;

  @Input() consultation: any = null;
  @Output() close = new EventEmitter<void>();

  fullDetail: any = null;
  reportData: any = null;
  summaryImageUrl: string | null = null;
  loading = false;

  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.loadDetail();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['consultation'] && changes['consultation'].currentValue) {
      this.loadDetail();
    }
  }

  loadDetail() {
    if (!this.consultation) return;

    this.loading = true;
    this.http.get<any>(`${this.API_BASE}/consultations/${this.consultation.id}`).subscribe({
      next: (data) => {
        this.fullDetail = data;
        this.loading = false;

        // Load scan report if session is linked
        if (data.session_id) {
          this.loadScanReport(data.session_id);
        }
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }

  loadScanReport(sessionId: string) {
    this.http.get<any>(`${this.API_BASE}/report/${sessionId}`).subscribe({
      next: (data) => {
        this.reportData = data;
        this.summaryImageUrl = `${this.API_BASE}/summary/${sessionId}?t=${Date.now()}`;
      },
      error: (err) => {
        console.error('Failed to load report:', err);
      }
    });
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

  formatDateTime(dateStr: string): string {
    if (!dateStr) return '—';
    const date = new Date(dateStr);
    return date.toLocaleDateString('fr-FR', {
      day: '2-digit', month: '2-digit', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  }
}
