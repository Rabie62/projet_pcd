import { Injectable } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';

export interface AppError {
  message: string;
  type: 'error' | 'warning' | 'info';
  timestamp: Date;
}

@Injectable({
  providedIn: 'root'
})
export class ErrorHandlerService {
  private errorsSubject = new BehaviorSubject<AppError[]>([]);
  public errors$: Observable<AppError[]> = this.errorsSubject.asObservable();

  private loadingSubject = new BehaviorSubject<boolean>(false);
  public loading$: Observable<boolean> = this.loadingSubject.asObservable();

  handleError(error: HttpErrorResponse | Error): void {
    let message = 'Une erreur inattendue est survenue.';
    let type: 'error' | 'warning' | 'info' = 'error';

    if (error instanceof HttpErrorResponse) {
      switch (error.status) {
        case 0:
          message = 'Impossible de se connecter au serveur. Vérifiez votre connexion.';
          break;
        case 400:
          message = error.error?.message || 'Requête invalide.';
          type = 'warning';
          break;
        case 401:
          message = 'Session expirée. Veuillez vous reconnecter.';
          break;
        case 403:
          message = 'Accès non autorisé.';
          type = 'warning';
          break;
        case 404:
          message = 'Ressource non trouvée.';
          type = 'warning';
          break;
        case 409:
          message = error.error?.message || 'Conflit de données.';
          type = 'warning';
          break;
        case 422:
          message = 'Données invalides. Veuillez vérifier le formulaire.';
          type = 'warning';
          break;
        case 500:
          message = 'Erreur serveur. Veuillez réessayer plus tard.';
          break;
        case 503:
          message = 'Service temporairement indisponible.';
          break;
        default:
          message = error.error?.message || `Erreur ${error.status}`;
      }
    } else if (error instanceof Error) {
      message = error.message;
    }

    this.addError(message, type);
  }

  addError(message: string, type: 'error' | 'warning' | 'info' = 'error'): void {
    const errors = this.errorsSubject.value;
    errors.push({ message, type, timestamp: new Date() });
    this.errorsSubject.next(errors);

    // Auto-remove after 5 seconds
    setTimeout(() => this.removeError(errors.length - 1), 5000);
  }

  removeError(index: number): void {
    const errors = this.errorsSubject.value;
    errors.splice(index, 1);
    this.errorsSubject.next([...errors]);
  }

  clearErrors(): void {
    this.errorsSubject.next([]);
  }

  setLoading(loading: boolean): void {
    this.loadingSubject.next(loading);
  }
}
