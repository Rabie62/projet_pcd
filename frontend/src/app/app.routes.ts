import { Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
  {
    path: 'login',
    loadComponent: () => import('./components/login/login.component').then(m => m.LoginComponent)
  },
  {
    path: 'register',
    loadComponent: () => import('./components/register/register.component').then(m => m.RegisterComponent)
  },
  {
    path: '',
    canActivate: [authGuard],
    loadComponent: () => import('./components/patient-dashboard/patient-dashboard.component').then(m => m.PatientDashboardComponent)
  },
  {
    path: 'analysis',
    canActivate: [authGuard],
    loadComponent: () => import('./components/scan-analysis/scan-analysis.component').then(m => m.ScanAnalysisComponent)
  },
  {
    path: 'chat',
    canActivate: [authGuard],
    loadComponent: () => import('./components/aichat-panel/aichat-panel.component').then(m => m.AIChatPanelComponent)
  },
  {
    path: 'knowledge',
    canActivate: [authGuard],
    loadComponent: () => import('./components/knowledge-base/knowledge-base.component').then(m => m.KnowledgeBaseComponent)
  },
  {
    path: 'medecins',
    canActivate: [authGuard],
    loadComponent: () => import('./components/medecin-dashboard/medecin-dashboard.component').then(m => m.MedecinDashboardComponent)
  },
  {
    path: 'consultations',
    canActivate: [authGuard],
    loadComponent: () => import('./components/consultation-dashboard/consultation-dashboard.component').then(m => m.ConsultationDashboardComponent)
  },
  {
    path: '**',
    redirectTo: ''
  }
];
