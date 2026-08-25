import { Component, ElementRef, ViewChild, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { LucideAngularModule, Brain, UploadCloud, FileImage, X, Play, CheckCircle, AlertTriangle, Activity, Download, Layers } from 'lucide-angular';
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
  readonly DownloadIcon = Download;
  readonly LayersIcon = Layers;

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

  exportPdf(): void {
    if (!this.result) return;

    const r = this.result;
    const printWindow = window.open('', '_blank', 'width=900,height=800');
    if (!printWindow) return;

    const tumorBadge = r.tumor_detected
      ? `<span style="background: #fee2e2; color: #dc2626; padding: 4px 12px; border-radius: 9999px; font-weight: bold; font-size: 12px;">ANOMALIE DÉTECTÉE</span>`
      : `<span style="background: #dcfce7; color: #16a34a; padding: 4px 12px; border-radius: 9999px; font-weight: bold; font-size: 12px;">AUCUNE ANOMALIE</span>`;

    const tumorSection = r.tumor_detected ? `
      <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <h3 style="margin: 0 0 8px 0; color: #0284c7;">Classification : ${r.classification}</h3>
        <p style="margin: 0 0 8px 0; color: #64748b; font-size: 14px;">
          Confiance IA : ${(r.classification_confidence * 100).toFixed(1)}% | Grade OMS : ${r.who_grade || 'Non spécifié'}
        </p>
        ${r.tumor_features ? `
          <div style="display: flex; gap: 24px; font-size: 14px; margin-top: 12px; border-top: 1px solid #e2e8f0; padding-top: 8px;">
            <div><strong>Surface :</strong> ${r.tumor_features.tumor_area_mm2?.toFixed(1) || '0'} mm²</div>
            <div><strong>Diamètre Max :</strong> ${r.tumor_features.max_diameter_mm?.toFixed(1) || '0'} mm</div>
            <div><strong>Localisation :</strong> ${r.tumor_features.location_description || 'Région centrale'}</div>
          </div>
        ` : ''}
      </div>
    ` : '';

    const imageSection = this.summaryImageUrl ? `
      <div style="margin-top: 24px; text-align: center;">
        <h3 style="text-align: left; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 16px;">Imagerie Médicale & Explicabilité IA</h3>
        <img src="${this.summaryImageUrl}" style="max-width: 100%; border-radius: 6px; margin-top: 12px; border: 1px solid #cbd5e1;" />
      </div>
    ` : '';

    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Rapport Médical - Patient ${r.patient_id}</title>
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b; padding: 40px; margin: 0; background: #fff; }
          .header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0284c7; padding-bottom: 16px; margin-bottom: 24px; }
          .logo { font-size: 20px; font-weight: bold; color: #0284c7; }
          .meta { font-size: 13px; color: #64748b; text-align: right; }
          .summary { white-space: pre-wrap; line-height: 1.6; font-size: 14px; color: #334155; }
          @media print {
            body { padding: 20px; }
            @page { margin: 15mm; size: A4; }
          }
        </style>
      </head>
      <body>
        <div class="header">
          <div>
            <div class="logo">Plateforme Médicale IA — Service de Neuroradiologie</div>
            <div style="font-size: 14px; color: #475569; margin-top: 4px;">Rapport d'Analyse IRM Cérébrale</div>
          </div>
          <div class="meta">
            <div><strong>Patient ID :</strong> ${r.patient_id}</div>
            <div><strong>Date :</strong> ${new Date().toLocaleDateString('fr-FR')} ${new Date().toLocaleTimeString('fr-FR')}</div>
          </div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
          <h2 style="margin: 0; font-size: 18px;">Diagnostic et Observations</h2>
          ${tumorBadge}
        </div>

        ${tumorSection}

        <div style="margin-top: 20px;">
          <h3 style="color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 16px; margin-bottom: 12px;">Résumé Clinique</h3>
          <div class="summary">${r.clinical_summary}</div>
        </div>

        ${imageSection}

        <script>
          window.onload = function() {
            setTimeout(function() {
              window.print();
            }, 500);
          };
        </script>
      </body>
      </html>
    `;

    printWindow.document.write(htmlContent);
    printWindow.document.close();
  }
}
