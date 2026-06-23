import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { Patient, PatientCreateRequest } from '../models';

@Injectable({
  providedIn: 'root'
})
export class PatientService {
  private apiUrl = `${environment.apiUrl}/patients`;

  constructor(private http: HttpClient) {}

  getPatients(page = 1, limit = 20): Observable<Patient[]> {
    const params = new HttpParams()
      .set('page', page.toString())
      .set('limit', limit.toString());
    return this.http.get<Patient[]>(this.apiUrl, { params, withCredentials: true });
  }

  getPatient(id: number): Observable<Patient> {
    return this.http.get<Patient>(`${this.apiUrl}/${id}`, { withCredentials: true });
  }

  createPatient(patient: PatientCreateRequest): Observable<Patient> {
    return this.http.post<Patient>(this.apiUrl, patient, { withCredentials: true });
  }

  updatePatient(id: number, patient: Partial<PatientCreateRequest>): Observable<Patient> {
    return this.http.put<Patient>(`${this.apiUrl}/${id}`, patient, { withCredentials: true });
  }

  deletePatient(id: number): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${id}`, { withCredentials: true });
  }

  getPatientHistory(id: number): Observable<string> {
    return this.http.get(`${this.apiUrl}/${id}/history`, {
      responseType: 'text',
      withCredentials: true
    });
  }
}
