import { Routes } from '@angular/router';
import { ScanAnalysisComponent } from './components/scan-analysis/scan-analysis.component';
import { PatientDashboardComponent } from './components/patient-dashboard/patient-dashboard.component';
import { AIChatPanelComponent } from './components/aichat-panel/aichat-panel.component';
import { KnowledgeBaseComponent } from './components/knowledge-base/knowledge-base.component';
import { MedecinDashboardComponent } from './components/medecin-dashboard/medecin-dashboard.component';
import { ConsultationDashboardComponent } from './components/consultation-dashboard/consultation-dashboard.component';
import { LoginComponent } from './components/login/login.component';
import { RegisterComponent } from './components/register/register.component';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
    { path: 'login', component: LoginComponent },
    { path: 'register', component: RegisterComponent },
    { path: '', component: PatientDashboardComponent, canActivate: [authGuard] },
    { path: 'analysis', component: ScanAnalysisComponent, canActivate: [authGuard] },
    { path: 'chat', component: AIChatPanelComponent, canActivate: [authGuard] },
    { path: 'knowledge', component: KnowledgeBaseComponent, canActivate: [authGuard] },
    { path: 'medecins', component: MedecinDashboardComponent, canActivate: [authGuard] },
    { path: 'consultations', component: ConsultationDashboardComponent, canActivate: [authGuard] },
    { path: '**', redirectTo: '' }
];
