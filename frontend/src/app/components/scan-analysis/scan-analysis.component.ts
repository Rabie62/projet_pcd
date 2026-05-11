import { Component, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Brain, UploadCloud, FileImage, X, Play, CheckCircle, AlertTriangle, Activity } from 'lucide-angular';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-scan-analysis',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './scan-analysis.component.html'
})
export class ScanAnalysisComponent {
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
  previewUrl: string | null = null;
  loading = false;
  result: any = null;
  error: string | null = null;
  patientId: any = '';
  viewMode: 'summary' = 'summary';

  @ViewChild('fileInput') fileInput!: ElementRef;

  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

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
    const validExts = ['.jpg', '.jpeg', '.png'];
    const filename = selectedFile.name.toLowerCase();
    const isValid = validExts.some(ext => filename.endsWith(ext));
    
    if (!isValid) {
      this.error = 'Veuillez télécharger un fichier de type JPG ou PNG.';
      return;
    }
    this.file = selectedFile;
    this.error = null;
    this.result = null;

    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    // Create a local preview for the image
    this.previewUrl = URL.createObjectURL(selectedFile);
  }

  clearFile() {
    this.file = null;
    if (this.previewUrl) URL.revokeObjectURL(this.previewUrl);
    this.previewUrl = null;
    this.error = null;
    this.result = null;
    if (this.fileInput?.nativeElement) {
      this.fileInput.nativeElement.value = '';
    }
  }

  handleAnalyze() {
    if (!this.file) return;
    if (this.patientId === null || this.patientId === undefined || this.patientId === '') {
      this.error = "L'ID du patient est requis pour lier l'historique d'examens.";
      return;
    }

    this.loading = true;
    this.error = null;
    this.result = null;

    const formData = new FormData();
    formData.append('file', this.file);
    formData.append('patient_id', this.patientId.toString());

    this.http.post(`${this.API_BASE}/analyze/upload`, formData).subscribe({
      next: (res: any) => {
        this.result = res;
        this.loading = false;
      },
      error: (err) => {
        console.error('API Error:', err);
        let parsedDetail = err.error?.detail || err.message;
        if(typeof parsedDetail === 'string' && parsedDetail.startsWith('{')) {
          try {
            parsedDetail = JSON.parse(parsedDetail).detail || parsedDetail;
          } catch(e) {}
        }
        this.error = parsedDetail || "L'analyse a échoué.";
        this.loading = false;
      }
    });
  }

  getSummaryUrl(): string {
    return `${this.API_BASE}/summary/${this.result.session_id}?t=${Date.now()}`;
  }

  getMathSize(size: number) {
     return (size / 1024 / 1024).toFixed(2);
  }
}

