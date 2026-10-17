import { Component, ViewChild, ElementRef, AfterViewChecked, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule, Send, Bot, User, Plus, MessageSquare, Trash2 } from 'lucide-angular';
import { marked } from 'marked';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { environment } from '../../../environments/environment';
import { AuthService } from '../../services/auth.service';

interface Message {
  role: 'assistant' | 'user';
  text: string;
}

interface ChatSession {
  id: string;
  title: string;
  updatedAt: string;
  messages: Message[];
}

@Component({
  selector: 'app-aichat-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule],
  templateUrl: './aichat-panel.component.html',
  styleUrls: ['./aichat-panel.component.css']
})
export class AIChatPanelComponent implements OnInit, AfterViewChecked {
  @ViewChild('scrollContainer') private scrollContainer!: ElementRef;

  readonly SendIcon = Send;
  readonly BotIcon = Bot;
  readonly UserIcon = User;
  readonly PlusIcon = Plus;
  readonly ChatIcon = MessageSquare;
  readonly TrashIcon = Trash2;

  conversations: ChatSession[] = [];
  activeConvId = '';
  sessionId = '';
  messages: Message[] = [];
  input = '';
  loading = false;

  private API_BASE = environment.apiUrl;

  constructor(
    private http: HttpClient,
    private sanitizer: DomSanitizer,
    private authService: AuthService
  ) {
    marked.setOptions({
      breaks: true,
      gfm: true
    });
  }

  ngOnInit(): void {
    this.loadConversations();
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    try {
      this.scrollContainer.nativeElement.scrollTop = this.scrollContainer.nativeElement.scrollHeight;
    } catch (err) {}
  }

  loadConversations(): void {
    try {
      const saved = localStorage.getItem('ai_chat_sessions');
      if (saved) {
        this.conversations = JSON.parse(saved);
      }
    } catch (err) {}

    if (this.conversations.length > 0) {
      this.selectConversation(this.conversations[0]);
    } else {
      this.newConversation();
    }

    this.http.get<any[]>(`${this.API_BASE}/sessions`).subscribe({
      next: (sessions) => {
        if (!sessions || sessions.length === 0) return;
        let added = false;
        for (const s of sessions) {
          const exists = this.conversations.some(c => c.id === s.session_id);
          if (!exists) {
            this.conversations.push({
              id: s.session_id,
              title: `Session ${s.session_id.substring(0, 8)}`,
              updatedAt: new Date().toISOString(),
              messages: [
                { role: 'assistant', text: `Session ${s.session_id} chargée. Vous pouvez poser vos questions sur ce scan.` }
              ]
            });
            added = true;
          }
        }
        if (added) this.saveConversations();
      },
      error: () => {}
    });
  }

  newConversation(): void {
    const id = 'chat-' + Math.random().toString(36).substring(2, 9);
    const conv: ChatSession = {
      id,
      title: 'Nouvelle conversation',
      updatedAt: new Date().toISOString(),
      messages: [
        { role: 'assistant', text: 'Bonjour Docteur. Je suis votre assistant médical IA en neuroradiologie. Comment puis-je vous aider ?' }
      ]
    };
    this.conversations.unshift(conv);
    this.selectConversation(conv);
    this.saveConversations();
  }

  selectConversation(conv: ChatSession): void {
    this.activeConvId = conv.id;
    this.sessionId = conv.id;
    this.messages = conv.messages;
    if (this.messages.length <= 1) {
      this.fetchBackendHistory(conv.id);
    }
  }

  deleteConversation(id: string, ev: Event): void {
    ev.stopPropagation();
    this.conversations = this.conversations.filter(c => c.id !== id);
    this.saveConversations();
    if (this.activeConvId === id) {
      if (this.conversations.length > 0) {
        this.selectConversation(this.conversations[0]);
      } else {
        this.newConversation();
      }
    }
  }

  saveConversations(): void {
    try {
      localStorage.setItem('ai_chat_sessions', JSON.stringify(this.conversations));
    } catch (err) {}
  }

  fetchBackendHistory(sid: string): void {
    this.http.get<any>(`${this.API_BASE}/chat/history/${sid}`).subscribe({
      next: (res) => {
        if (res && res.history && res.history.length > 0) {
          const loaded: Message[] = [];
          for (const item of res.history) {
            loaded.push({ role: 'user', text: item.user });
            loaded.push({ role: 'assistant', text: item.assistant });
          }
          this.messages = loaded;
          const cur = this.conversations.find(c => c.id === sid);
          if (cur) {
            cur.messages = loaded;
            this.saveConversations();
          }
        }
      },
      error: () => {}
    });
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

    const cur = this.conversations.find(c => c.id === this.activeConvId);
    if (cur) {
      cur.messages = this.messages;
      cur.updatedAt = new Date().toISOString();
      if (cur.title === 'Nouvelle conversation') {
        cur.title = userMsg.substring(0, 26) + (userMsg.length > 26 ? '...' : '');
      }
      this.saveConversations();
    }

    const assistantMsg: Message = { role: 'assistant', text: '' };
    this.messages.push(assistantMsg);

    const payload: any = { query: userMsg };
    if (this.sessionId.trim()) payload.session_id = this.sessionId.trim();

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    const token = this.authService.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    try {
      const response = await fetch(`${this.API_BASE}/chat/stream`, {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Connection Error: Unable to reach Dialogue Agent.');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Stream reader not available');

      this.loading = false;
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        assistantMsg.text += chunk;
      }

      if (cur) {
        cur.messages = this.messages;
        this.saveConversations();
      }
    } catch (err: any) {
      assistantMsg.text = err.message || 'Une erreur inattendue est survenue.';
      this.loading = false;
      if (cur) {
        cur.messages = this.messages;
        this.saveConversations();
      }
    }
  }
}
