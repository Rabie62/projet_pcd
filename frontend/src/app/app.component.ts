import { Component } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';
import { LucideAngularModule, Activity, Users, FileSearch, MessageSquare, Database, Stethoscope, ClipboardList } from 'lucide-angular';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, LucideAngularModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'Medical AI Hub';
  
  readonly ActivityIcon = Activity;
  readonly UsersIcon = Users;
  readonly FileSearchIcon = FileSearch;
  readonly MessageSquareIcon = MessageSquare;
  readonly DatabaseIcon = Database;
  readonly StethoscopeIcon = Stethoscope;
  readonly ClipboardListIcon = ClipboardList;
}
