import { Component, ElementRef, ViewChild, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { LucideAngularModule, Brain, UploadCloud, FileImage, X, Play, CheckCircle, AlertTriangle, Activity } from 'lucide-angular';
import { Subject, takeUntil } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ErrorHandlerService } from '../../services/error-handler.service';
import { AnalysisResponse } from '../../models';

@Component({
  selector: 'app-scan-analysis',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './scan-analysis.component.html',
  styleUrls: ['./scan-analysis.component.css']
})
export class ScanAnalysisComponent implements OnDestroy {
  readonly BrainIcon = Brain;
  readonly UploadCloudIcon = UploadCloud;
  readonly FileImageIcon = FileImage;
  readonly XIcon = X;
  readonly PlayIcon = Play;
  readonly CheckCircleIcon = CheckCircle;
  readonly AlertTriangleIcon = AlertTriangle;
  readonly ActivityIcon = Activity;

  file: File | null = null;
  isDragging = false;
  previewUrl: SafeUrl | string | null = null;
  summaryImageUrl: string | null = null;
  private rawPreviewUrl: string | null = null;
  loading = false;
  result: AnalysisResponse | null = null;
  error: string | null = null;
  patientId: number | null = null;

  private destroy$ = new Subject<void>();
  private readonly MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
  private readonly ALLOWED_TYPES = ['image/jpeg', 'image/png'];

  @ViewChild('fileInput') fileInput!: ElementRef;

  private API_BASE = environment.apiUrl;

  constructor(
    private http: HttpClient,
    private sanitizer: DomSanitizer,
    private errorHandler: ErrorHandlerService
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.rawPreviewUrl) URL.revokeObjectURL(this.rawPreviewUrl);
  }

  handleDragOver(e: DragEvent) {
    e.preventDefault();
    this.isDragging = true;
  }

  handleDragLeave(e: DragEvent) {
    e.preventDefault();
    this.isDragging = false;
  }

  handleDrop(e: DragEvent) {
    e.preventDefault();
    this.isDragging = false;
    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
      this.handleFileSelection(e.dataTransfer.files[0]);
    }
  }

  handleFileChange(e: Event) {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      this.handleFileSelection(target.files[0]);
    }
  }

  handleFileSelection(selectedFile: File) {
    // Validate file type
    if (!this.ALLOWED_TYPES.includes(selectedFile.type)) {
      this.error = 'Veuillez télécharger un fichier de type JPG ou PNG.';
      return;
    }

    // Validate file size
    if (selectedFile.size > this.MAX_FILE_SIZE) {
      this.error = `Le fichier est trop volumineux (${this.getMathSize(selectedFile.size)}). Maximum: ${this.getMathSize(this.MAX_FILE_SIZE)}.`;
      return;
    }

    this.file = selectedFile;
    this.error = null;
    this.result = null;

    if (this.rawPreviewUrl) URL.revokeObjectURL(this.rawPreviewUrl);
    this.rawPreviewUrl = URL.createObjectURL(selectedFile);
    this.previewUrl = this.sanitizer.bypassSecurityTrustUrl(this.rawPreviewUrl);
  }

  clearFile() {
    this.file = null;
    if (this.rawPreviewUrl) URL.revokeObjectURL(this.rawPreviewUrl);
    this.rawPreviewUrl = null;
    this.previewUrl = null;
    this.summaryImageUrl = null;
    this.error = null;
    this.result = null;
    if (this.fileInput?.nativeElement) {
      this.fileInput.nativeElement.value = '';
    }
  }

  handleAnalyze() {
    if (!this.file) return;
    if (this.patientId === null || this.patientId === undefined || this.patientId <= 0) {
      this.error = "L'ID du patient est requis pour lier l'historique d'examens.";
      return;
    }

    this.loading = true;
    this.error = null;
    this.result = null;
    this.summaryImageUrl = null;
    this.errorHandler.setLoading(true);

    const formData = new FormData();
    formData.append('file', this.file);
    formData.append('patient_id', this.patientId.toString());

    this.http.post<AnalysisResponse>(`${this.API_BASE}/analyze/upload`, formData, {
      withCredentials: true
    }).pipe(takeUntil(this.destroy$)).subscribe({
      next: (res) => {
        this.result = res;
        this.summaryImageUrl = `${this.API_BASE}/summary/${res.session_id}?t=${Date.now()}`;
        this.loading = false;
        this.errorHandler.setLoading(false);
      },
      error: (err) => {
        this.errorHandler.handleError(err);
        this.loading = false;
        this.errorHandler.setLoading(false);
      }
    });
  }

  getMathSize(size: number): string {
    return (size / 1024 / 1024).toFixed(2);
  }
}
