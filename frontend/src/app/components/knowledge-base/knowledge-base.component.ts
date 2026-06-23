import { Component, OnInit, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { LucideAngularModule, UploadCloud, FileText, Trash2 } from 'lucide-angular';
import { firstValueFrom, Subject, takeUntil } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ErrorHandlerService } from '../../services/error-handler.service';
import { KnowledgeDocument, KnowledgeStatus } from '../../models';

@Component({
  selector: 'app-knowledge-base',
  standalone: true,
  imports: [CommonModule, LucideAngularModule],
  templateUrl: './knowledge-base.component.html',
  styleUrls: ['./knowledge-base.component.css']
})
export class KnowledgeBaseComponent implements OnInit, OnDestroy {
  readonly UploadCloudIcon = UploadCloud;
  readonly FileTextIcon = FileText;
  readonly Trash2Icon = Trash2;

  docs: KnowledgeDocument[] = [];
  loading = true;
  status: KnowledgeStatus = { available: false, total_chunks: 0, uploaded_documents: 0, system_knowledge_indexed: false };
  uploading = false;

  private destroy$ = new Subject<void>();
  private API_BASE = environment.apiUrl;
  private readonly MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
  private readonly ALLOWED_TYPES = ['.txt', '.md', '.pdf'];

  constructor(
    private http: HttpClient,
    private errorHandler: ErrorHandlerService
  ) {}

  ngOnInit(): void {
    this.fetchDocs();
    this.fetchStatus();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  async fetchDocs(): Promise<void> {
    try {
      this.docs = await firstValueFrom(
        this.http.get<KnowledgeDocument[]>(`${this.API_BASE}/knowledge/documents`, { withCredentials: true })
          .pipe(takeUntil(this.destroy$))
      );
    } catch (err) {
      this.errorHandler.handleError(err as any);
    }
  }

  async fetchStatus(): Promise<void> {
    try {
      this.status = await firstValueFrom(
        this.http.get<KnowledgeStatus>(`${this.API_BASE}/knowledge/status`, { withCredentials: true })
          .pipe(takeUntil(this.destroy$))
      );
      this.loading = false;
    } catch (err) {
      this.errorHandler.handleError(err as any);
      this.loading = false;
    }
  }

  async handleUpload(e: Event, fileInput: HTMLInputElement): Promise<void> {
    e.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) return;

    // Validate file type
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!this.ALLOWED_TYPES.includes(ext)) {
      this.errorHandler.addError(`Type de fichier non supporté. Formats acceptés: ${this.ALLOWED_TYPES.join(', ')}`, 'warning');
      return;
    }

    // Validate file size
    if (file.size > this.MAX_FILE_SIZE) {
      this.errorHandler.addError(`Fichier trop volumineux (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum: 10 MB.`, 'warning');
      return;
    }

    this.uploading = true;
    this.errorHandler.setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await firstValueFrom(
        this.http.post(
          `${this.API_BASE}/knowledge/upload`,
          formData,
          { withCredentials: true }
        ).pipe(takeUntil(this.destroy$))
      );
      await this.fetchDocs();
      await this.fetchStatus();
      this.errorHandler.addError('Document uploadé avec succès.', 'info');
    } catch (err) {
      this.errorHandler.handleError(err as any);
    } finally {
      this.uploading = false;
      this.errorHandler.setLoading(false);
      const target = e.target as HTMLFormElement;
      if (target) target.reset();
    }
  }

  async handleDelete(id: string): Promise<void> {
    if (!window.confirm('Supprimer ce document ? Cette action est irréversible.')) return;

    try {
      await firstValueFrom(
        this.http.delete(`${this.API_BASE}/knowledge/documents/${id}`, { withCredentials: true })
          .pipe(takeUntil(this.destroy$))
      );
      await this.fetchDocs();
      await this.fetchStatus();
      this.errorHandler.addError('Document supprimé.', 'info');
    } catch (err) {
      this.errorHandler.handleError(err as any);
    }
  }
}
