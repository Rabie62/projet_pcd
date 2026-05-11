import { Component, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Send, Bot, User } from 'lucide-angular';
import { marked } from 'marked';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { environment } from '../../../environments/environment';

interface Message {
  role: 'assistant' | 'user';
  text: string;
}

@Component({
  selector: 'app-aichat-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './aichat-panel.component.html',
  styleUrls: ['./aichat-panel.component.css']
})
export class AIChatPanelComponent implements AfterViewChecked {
  @ViewChild('scrollContainer') private scrollContainer!: ElementRef;

  readonly SendIcon = Send;
  readonly BotIcon = Bot;
  readonly UserIcon = User;

  messages: Message[] = [
    { role: 'assistant', text: 'Bonjour Docteur. Je suis votre assistant medical IA specialise en neuroradiologie. Comment puis-je vous aider avec l\'analyse des patients ?' }
  ];
  input = '';
  sessionId = '';
  loading = false;

  private API_BASE = environment.apiBase;

  constructor(private sanitizer: DomSanitizer) {
    // Configure marked options if needed
    marked.setOptions({
      breaks: true,
      gfm: true
    });
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    try {
      this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
    } catch (err) {}
  }

  renderMarkdown(text: string): SafeHtml {
    const html = marked.parse(text) as string;
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  async sendMessage(e?: Event) {
    if (e) e.preventDefault();
    if (!this.input.trim()) return;

    const userMsg = this.input.trim();
    this.messages.push({ role: 'user', text: userMsg });
    this.input = '';
    this.loading = true;

    // Create a placeholder for the assistant response
    const assistantMsg: Message = { role: 'assistant', text: '' };
    this.messages.push(assistantMsg);

    const payload: any = { query: userMsg };
    if (this.sessionId.trim()) payload.session_id = this.sessionId.trim();

    try {
      const response = await fetch(`${this.API_BASE}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Connection Error: Unable to reach Dialogue Agent.');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Stream reader not available');

      // Stop loading spinner once we have a valid response stream
      this.loading = false;

      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        assistantMsg.text += chunk;
      }
    } catch (err: any) {
      assistantMsg.text = err.message || 'An unexpected error occurred.';
      this.loading = false;
    }
  }
}
