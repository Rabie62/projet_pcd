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
    const savedUser = localStorage.getItem('auth_user');
    if (savedUser) {
      try {
        const user: AuthUser = JSON.parse(savedUser);
        this.currentUserSubject.next(user);
      } catch (e) {
        localStorage.removeItem('auth_user');
      }
    }
  }

  login(credentials: { username: string; password: string }): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/login`, credentials, {
      withCredentials: true
    }).pipe(
      tap(response => {
          if (response.token) {
            localStorage.setItem('auth_token', response.token);
          }
          const user: AuthUser = {
            id: response.id,
            nom: response.nom,
            prenom: response.prenom,
            email: response.email,
            specialite: response.specialite,
            departement: response.departement,
            username: response.username,
          };
          localStorage.setItem('auth_user', JSON.stringify(user));
          this.currentUserSubject.next(user);
      }),
      catchError(err => this.handleError(err))
    );
  }

  register(userData: any): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/register`, userData, {
      withCredentials: true
    }).pipe(
      tap(response => {
        if (response) {
          if (response.token) {
            localStorage.setItem('auth_token', response.token);
          }
          const user: AuthUser = {
            id: response.id,
            nom: response.nom,
            prenom: response.prenom,
            email: response.email,
            specialite: response.specialite,
            departement: response.departement,
            username: response.username,
          };
          localStorage.setItem('auth_user', JSON.stringify(user));
          this.currentUserSubject.next(user);
        }
      }),
      catchError(err => this.handleError(err))
    );
  }

  logout(): void {
    this.http.post(`${this.apiUrl}/logout`, {}, { withCredentials: true }).subscribe({
      next: () => this.clearAuth(),
      error: () => this.clearAuth(),
    });
  }

  private clearAuth(): void {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    this.currentUserSubject.next(null);
    this.router.navigate(['/login']);
  }

  getToken(): string | null {
    return localStorage.getItem('auth_token');
  }

  isLoggedIn(): boolean {
    return !!this.currentUserSubject.value;
  }

  getCurrentUser(): AuthUser | null {
    return this.currentUserSubject.value;
  }

  getUserDisplayName(): string {
    const user = this.currentUserSubject.value;
    if (!user) return 'Docteur';
    if (user.prenom && user.nom) {
      return `Dr. ${user.prenom} ${user.nom}`;
    }
    if (user.nom) {
      return `Dr. ${user.nom}`;
    }
    if (user.username) {
      return `Dr. ${user.username}`;
    }
    return 'Docteur';
  }

  getUserInitials(): string {
    const user = this.currentUserSubject.value;
    if (!user) return 'DR';
    const first = user.prenom?.trim()?.[0] || '';
    const last = user.nom?.trim()?.[0] || '';
    const combined = (first + last).toUpperCase();
    if (combined) return combined;
    if (user.username) return user.username.substring(0, 2).toUpperCase();
    return 'DR';
  }

  getUserRole(): string {
    const user = this.currentUserSubject.value;
    if (!user) return 'Praticien';
    return user.specialite || user.departement || 'Praticien';
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
