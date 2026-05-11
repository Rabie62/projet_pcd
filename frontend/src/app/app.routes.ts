import { Routes } from '@angular/router';
import { ScanAnalysisComponent } from './components/scan-analysis/scan-analysis.component';
import { PatientDashboardComponent } from './components/patient-dashboard/patient-dashboard.component';
import { AIChatPanelComponent } from './components/aichat-panel/aichat-panel.component';
import { KnowledgeBaseComponent } from './components/knowledge-base/knowledge-base.component';
import { MedecinDashboardComponent } from './components/medecin-dashboard/medecin-dashboard.component';
import { ConsultationDashboardComponent } from './components/consultation-dashboard/consultation-dashboard.component';

export const routes: Routes = [
    { path: '', component: PatientDashboardComponent },
    { path: 'analysis', component: ScanAnalysisComponent },
    { path: 'chat', component: AIChatPanelComponent },
    { path: 'knowledge', component: KnowledgeBaseComponent },
    { path: 'medecins', component: MedecinDashboardComponent },
    { path: 'consultations', component: ConsultationDashboardComponent }
];
