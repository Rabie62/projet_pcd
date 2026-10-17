"""
Dialogue Agent — Ollama natural language interface.
Communicates diagnostic results to doctors and answers medical queries.
"""

from __future__ import annotations
import json
import re
from typing import Optional
import httpx
from loguru import logger
from config.settings import Settings, get_settings
from agents.diagnostic import DiagnosticReport


class DialogueAgent:
    """
    Dialogue Agent using Ollama.

    Provides:
    - Natural language explanations of diagnostic findings
    - Answers to doctor queries about the analysis
    - Confidence scores and medical disclaimers
    - Evidence-based responses grounded in the diagnostic report
    """

    # ── System prompt (directive, clinical, comprehensive) ──
    SYSTEM_PROMPT = (
        "Assistant medical IA en neuroradiologie. Repondez en francais de maniere complete et detaillee.\n"
        "Regles :\n"
        "- Basez vos reponses sur le rapport de diagnostic fourni et les connaissances medicales\n"
        "- Citez les mesures, la classification et le grade OMS\n"
        "- Mentionnez la confiance IA pour chaque resultat\n"
        "- Proposez des protocoles medicaux concrets (chirurgie, suivi, bilan, diagnostic differentiel)\n"
        "- Fournissez des explications approfondies et exhaustives\n"
        "- Terminez par : 'Confirmation clinique par un medecin requise.'"
    )

    # ── Few-shot examples (anchor the expected response style) ──
    CHAT_EXAMPLES = [
        {
            "user": "Quelle est la classification detectee ?",
            "assistant": (
                "Un gliome a ete detecte avec une confiance IA de 93.2%. "
                "Le grade OMS estime est II-IV, necessitant une confirmation histopathologique. "
                "La tumeur mesure 42.1 mm de diametre maximal dans la region anterieure droite.\n"
                "Confirmation clinique par un medecin requise."
            ),
        },
        {
            "user": "Quel est le grade et que recommandez-vous ?",
            "assistant": (
                "Grade OMS estime : II-IV. Le gliome est un processus agressif. "
                "Recommandations : consultation neurochirurgicale urgente, IRM de perfusion "
                "pour evaluation precise du grade, et eventuelle biopsie stereotaxique.\n"
                "Confirmation clinique par un medecin requise."
            ),
        },
        {
            "user": "La confiance est-elle suffisante ?",
            "assistant": (
                "La confiance IA est de 78.5%, ce qui est en dessous du seuil de 85%. "
                "Un diagnostic differentiel doit etre envisage (meningiome, metastase). "
                "Je recommande une verification histopathologique et une imagerie complementaire.\n"
                "Confirmation clinique par un medecin requise."
            ),
        },
    ]

    def __init__(self, settings: Optional[Settings] = None, rag_system=None):
        self.settings = settings or get_settings()
        self.rag_system = rag_system
        self.base_url = self.settings.model.ollama_base_url.rstrip("/")
        self.model_name = self.settings.model.ollama_model
        self.model_loaded = False
        self.current_report: Optional[DiagnosticReport] = None

        # Compatibility placeholders
        self.model = None
        self.tokenizer = None
        
        # Per-session multi-turn memory: session_id -> list of {"user": str, "assistant": str}
        self.session_memories: dict[str, list[dict[str, str]]] = {}
        # Per-session diagnostic reports: session_id -> DiagnosticReport
        self.session_reports: dict[str, DiagnosticReport] = {}

    def load_model(self) -> None:
        """Verify Ollama connection and model availability."""
        logger.info(f"Connecting to Ollama at {self.base_url} (model: {self.model_name})...")
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name", "") for m in res.json().get("models", [])]
                    # Check exact or prefix match (e.g. 'llama3.2:3b' or 'llama3.2:3b-instruct')
                    found = any(self.model_name in m or m.startswith(self.model_name) for m in models)
                    if found:
                        logger.info(f"Ollama model '{self.model_name}' is ready.")
                    else:
                        logger.warning(
                            f"Model '{self.model_name}' not found in local Ollama list ({models}). "
                            f"Ollama will attempt to pull it or use default."
                        )
                    self.model_loaded = True
                else:
                    logger.warning(f"Ollama responded with HTTP {res.status_code}.")
                    self.model_loaded = True
        except Exception as e:
            logger.warning(f"Unable to reach Ollama at {self.base_url}: {e}. Inference will retry on demand.")
            self.model_loaded = False

    def set_report_context(self, report: DiagnosticReport, session_id: Optional[str] = None) -> None:
        """Set the diagnostic report as conversation context for a session."""
        self.current_report = report
        if session_id:
            self.session_reports[session_id] = report
        logger.debug(f"Set report context for patient {report.patient_id} (session: {session_id})")

    def get_session_report(self, session_id: Optional[str] = None) -> Optional[DiagnosticReport]:
        """Get the diagnostic report associated with a session."""
        if session_id and session_id in self.session_reports:
            return self.session_reports[session_id]
        return self.current_report

    def get_session_history(self, session_id: Optional[str] = None) -> list[dict[str, str]]:
        """Get chat history for a session."""
        key = session_id or "default"
        return self.session_memories.get(key, [])

    def add_session_exchange(self, session_id: Optional[str], user_query: str, assistant_response: str) -> None:
        """Add a conversation turn to session memory."""
        key = session_id or "default"
        if key not in self.session_memories:
            self.session_memories[key] = []
        self.session_memories[key].append({
            "user": user_query,
            "assistant": assistant_response,
        })
        # Keep up to the last 10 exchanges per session to avoid token overflow
        if len(self.session_memories[key]) > 10:
            self.session_memories[key] = self.session_memories[key][-10:]

    def clear_session_history(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history for a specific session or all sessions."""
        if session_id:
            self.session_memories.pop(session_id, None)
        else:
            self.session_memories.clear()

    def build_context_prompt(self, session_id: Optional[str] = None) -> str:
        """Build a structured context string from the diagnostic report and RAG."""
        report = self.get_session_report(session_id)
        if report is None:
            return "Aucun rapport de diagnostic disponible."

        parts = [
            f"Patient : {report.patient_id}",
        ]

        if report.tumor_detected:
            parts.append("Tumeur detectee : Oui")
            if report.tumor_features:
                f = report.tumor_features
                parts.extend([
                    f"Classification : {report.classification} "
                    f"(confiance : {report.classification_confidence:.1%})",
                    f"Grade OMS : {report.who_grade}",
                    f"Surface : {f.tumor_area_mm2:.1f} mm2",
                    f"Diametre max : {f.max_diameter_mm:.1f} mm",
                    f"Localisation : {f.location_description}",
                    f"Ratio tumoral : {f.tumor_ratio:.1%}",
                ])
            if report.flags:
                parts.append(f"Alertes : {'; '.join(report.flags)}")
            if report.recommendations:
                parts.append(
                    f"Recommandations : {'; '.join(report.recommendations)}"
                )
            # Include full clinical summary
            if report.clinical_summary:
                parts.append(f"Resume clinique : {report.clinical_summary}")
        else:
            parts.append("Tumeur detectee : Non")
            parts.append("Examen normal.")

        # ── RAG Context (full) ──
        if (self.rag_system and self.rag_system.available
                and report.tumor_detected and report.classification):
            classification = report.classification
            conf = report.classification_confidence
            queries = [
                f"{classification} prise en charge traitement",
                f"{classification} protocole medical recommandation",
            ]
            if conf < 0.85:
                queries.append(f"diagnostic differentiel {classification}")

            seen = set()
            rag_parts = []
            for q in queries:
                for chunk in self.rag_system.retrieve(q, top_k=4, min_score=0.25):
                    key = chunk.text[:80]
                    if key not in seen:
                        seen.add(key)
                        rag_parts.append(chunk.text)

            if rag_parts:
                rag_text = "\n---\n".join(rag_parts)
                parts.append(f"\nConnaissances cliniques :\n{rag_text}")

        return "\n".join(parts)

    def build_messages(self, user_query: str, session_id: Optional[str] = None) -> list[dict[str, str]]:
        """Build the chat messages list for Ollama's chat endpoint with session memory."""
        context = self.build_context_prompt(session_id)

        # ── System message: prompt + few-shot examples ──
        examples_text = ""
        for ex in self.CHAT_EXAMPLES:
            examples_text += f"\nMedecin : {ex['user']}\nAssistant : {ex['assistant']}\n"

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Exemples de reponses attendues :\n"
            f"{examples_text}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Voici le rapport de diagnostic du patient. Memorisez-le."},
            {"role": "assistant", "content": f"Rapport enregistre :\n{context}"},
        ]

        # ── Multi-turn conversation history for this session (last 4 exchanges) ──
        history = self.get_session_history(session_id)
        for exchange in history[-4:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        # ── Current query ──
        messages.append({"role": "user", "content": user_query})
        return messages

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat completion request to Ollama.

        Args:
            messages: list of {"role": str, "content": str}
            temperature: sampling temperature

        Returns:
            Generated response string
        """
        if not self.model_loaded:
            self.load_model()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.settings.model.llm_temperature,
            },
        }

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

    def chat(self, user_query: str, session_id: Optional[str] = None) -> str:
        """
        Process a doctor's query and generate a response with session memory.

        Args:
            user_query: natural language question from the doctor
            session_id: optional unique ID of the analysis session

        Returns:
            Natural language response grounded in the diagnostic report and chat history
        """
        messages = self.build_messages(user_query, session_id=session_id)
        response = self.generate_chat(messages)

        # Post-processing: strip leaked metadata lines
        response = re.sub(r'\[RAG Metrics:.*?\]', '', response).strip()
        response = re.sub(r'Requete RAG:.*?\n', '', response).strip()

        self.add_session_exchange(session_id, user_query, response)
        return response

    def chat_stream(self, user_query: str, session_id: Optional[str] = None):
        """
        Process a doctor's query and generate a streamed response with session memory.

        Args:
            user_query: natural language question from the doctor
            session_id: optional unique ID of the analysis session

        Yields:
            Text chunks as they are generated by Ollama
        """
        if not self.model_loaded:
            self.load_model()

        messages = self.build_messages(user_query, session_id=session_id)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.settings.model.llm_temperature,
            },
        }

        full_response = ""
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk_data = json.loads(line)
                        token = chunk_data.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            yield token
                        if chunk_data.get("done"):
                            break
                    except Exception as err:
                        logger.debug(f"Failed to parse streaming line: {err}")

        # Record conversation history into session memory
        cleaned = re.sub(r'\[RAG Metrics:.*?\]', '', full_response).strip()
        cleaned = re.sub(r'Requete RAG:.*?\n', '', cleaned).strip()
        self.add_session_exchange(session_id, user_query, cleaned)

    def generate_summary(self) -> str:
        """Generate a natural language summary of the diagnostic findings."""
        if self.current_report is None:
            return "Aucun rapport de diagnostic disponible."

        query = (
            "Resumez les resultats de l'IRM : localisation, taille, classification, "
            "confiance IA, et recommandations. Soyez concis."
        )
        return self.chat(query)

    def explain_finding(self, finding: str) -> str:
        """Explain a specific finding from the report in detail."""
        query = f"Expliquez en detail le resultat suivant : {finding}"
        return self.chat(query)

    def answer_location_query(self) -> str:
        """Answer 'Where is the tumor located?'"""
        return self.chat(
            "Ou se situe exactement la tumeur ? Decrivez la localisation "
            "anatomique et les structures voisines potentiellement atteintes."
        )

    def explain_confidence(self) -> str:
        """Explain the AI confidence scores and their clinical meaning."""
        return self.chat(
            "Expliquez le score de confiance de la classification tumorale. "
            "Que signifie-t-il cliniquement ? Est-il suffisant ?"
        )
