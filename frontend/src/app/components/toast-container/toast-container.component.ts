import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ErrorHandlerService, AppError } from '../../services/error-handler.service';

@Component({
  selector: 'app-toast-container',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="toast-container">
      <div
        *ngFor="let error of errors; let i = index"
        [class]="'toast toast-' + error.type"
        role="alert"
        aria-live="polite"
      >
        <span class="toast-message">{{ error.message }}</span>
        <button class="toast-close" (click)="remove(i)" aria-label="Fermer">×</button>
      </div>
    </div>
  `,
  styles: [`
    .toast-container {
      position: fixed;
      top: 1rem;
      right: 1rem;
      z-index: 2000;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      max-width: 400px;
    }
    .toast {
      padding: 1rem 1.5rem;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      animation: slideIn 0.3s ease;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .toast-error { background: rgba(239,68,68,0.9); color: white; }
    .toast-warning { background: rgba(245,158,11,0.9); color: white; }
    .toast-info { background: rgba(56,189,248,0.9); color: white; }
    .toast-close {
      background: none;
      border: none;
      color: inherit;
      font-size: 1.2rem;
      cursor: pointer;
      padding: 0 0.25rem;
    }
    @keyframes slideIn {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `]
})
export class ToastContainerComponent {
  errors: AppError[] = [];

  constructor(private errorHandler: ErrorHandlerService) {
    this.errorHandler.errors$.subscribe(errors => {
      this.errors = [...errors];
    });
  }

  remove(index: number): void {
    this.errorHandler.removeError(index);
  }
}
