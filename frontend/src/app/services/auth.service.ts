import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, BehaviorSubject, throwError } from 'rxjs';
import { tap, catchError } from 'rxjs/operators';
import { Router } from '@angular/router';
import { environment } from '../../environments/environment';

export interface AuthUser {
  id: number;
  nom: string;
  prenom: string;
  email: string | null;
  specialite: string | null;
  departement: string | null;
  username: string;
}

export interface AuthResponse {
  token: string;
  id: number;
  nom: string;
  prenom: string;
  email: string | null;
  specialite: string | null;
  departement: string | null;
  username: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = `${environment.apiUrl}/auth`;
  private currentUserSubject = new BehaviorSubject<AuthUser | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor(
    private http: HttpClient,
    private router: Router
  ) {
    // Note: JWT is stored in httpOnly cookie by server
    // We don't need to restore from cookie since browser sends it automatically
  }

  login(credentials: { username: string; password: string }): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/login`, credentials, {
      withCredentials: true
    }).pipe(
      tap(response => {
        if (response) {
          const user: AuthUser = {
            id: response.id,
            nom: response.nom,
            prenom: response.prenom,
            email: response.email,
            specialite: response.specialite,
            departement: response.departement,
            username: response.username,
          };
          this.currentUserSubject.next(user);
        }
      }),
      catchError(this.handleError)
    );
  }

  register(userData: any): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/register`, userData, {
      withCredentials: true
    }).pipe(
      tap(response => {
        if (response) {
          const user: AuthUser = {
            id: response.id,
            nom: response.nom,
            prenom: response.prenom,
            email: response.email,
            specialite: response.specialite,
            departement: response.departement,
            username: response.username,
          };
          this.currentUserSubject.next(user);
        }
      }),
      catchError(this.handleError)
    );
  }

  logout(): void {
    this.http.post(`${this.apiUrl}/logout`, {}, { withCredentials: true }).subscribe({
      next: () => this.clearAuth(),
      error: () => this.clearAuth(),
    });
  }

  private clearAuth(): void {
    this.currentUserSubject.next(null);
    this.router.navigate(['/login']);
  }

  isLoggedIn(): boolean {
    return !!this.currentUserSubject.value;
  }

  getCurrentUser(): AuthUser | null {
    return this.currentUserSubject.value;
  }

  private handleError(error: HttpErrorResponse): Observable<never> {
    let message = 'Une erreur est survenue.';
    if (error.status === 401) {
      message = 'Identifiants incorrects.';
    } else if (error.status === 409) {
      message = error.error?.message || 'Ce nom d\'utilisateur est déjà pris.';
    } else if (error.error?.message) {
      message = error.error.message;
    }
    return throwError(() => new Error(message));
  }
}
