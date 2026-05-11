import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, UploadCloud, FileText, Trash2 } from 'lucide-angular';
import { firstValueFrom } from 'rxjs';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-knowledge-base',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './knowledge-base.component.html'
})
export class KnowledgeBaseComponent implements OnInit {
  readonly UploadCloudIcon = UploadCloud;
  readonly FileTextIcon = FileText;
  readonly Trash2Icon = Trash2;

  docs: any[] = [];
  loading = true;
  status: any = {};
  uploading = false;

  private API_BASE = environment.apiBase;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchDocs();
    this.fetchStatus();
  }

  async fetchDocs() {
    try {
      this.docs = await firstValueFrom(this.http.get<any[]>(`${this.API_BASE}/knowledge/documents`));
    } catch(e) {}
  }

  async fetchStatus() {
    try {
      this.status = await firstValueFrom(this.http.get<any>(`${this.API_BASE}/knowledge/status`));
      this.loading = false;
    } catch(e) {}
  }

  uploadedBy: string = '';

  async handleUpload(e: Event, fileInput: HTMLInputElement) {
    e.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) return;

    const doctorName = this.uploadedBy.trim() || 'Anonyme';
    this.uploading = true;
    const formData = new FormData();
    formData.append('file', file);

    try {
      await firstValueFrom(this.http.post(`${this.API_BASE}/knowledge/upload?uploaded_by=${encodeURIComponent(doctorName)}`, formData));
      await this.fetchDocs();
      await this.fetchStatus();
    } catch(err) {
      alert("Upload failed.");
    } finally {
      this.uploading = false;
      const target = e.target as HTMLFormElement;
      if(target) target.reset();
    }
  }

  async handleDelete(id: string) {
    try {
      await firstValueFrom(this.http.delete(`${this.API_BASE}/knowledge/documents/${id}`));
      await this.fetchDocs();
      await this.fetchStatus();
    } catch(e) {}
  }
}
