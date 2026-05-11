"""
Dialogue Agent — Gemma 3 1B-IT natural language interface.
Communicates diagnostic results to doctors and answers medical queries.
"""

from __future__ import annotations
import re
from typing import Optional
import torch
from loguru import logger
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from config.settings import Settings, get_settings
from agents.diagnostic import DiagnosticReport


class DialogueAgent:
    """
    Dialogue Agent using Gemma 3 1B-IT.

    Provides:
    - Natural language explanations of diagnostic findings
    - Answers to doctor queries about the analysis
    - Confidence scores and medical disclaimers
    - Evidence-based responses grounded in the diagnostic report
    """

    # ── System prompt (concise, directive, optimized for 1B) ──
    SYSTEM_PROMPT = (
        "Assistant medical IA en neuroradiologie. Repondez en francais.\n"
        "Regles :\n"
        "- Basez vos reponses uniquement sur le rapport de diagnostic fourni\n"
        "- Citez les mesures, la classification et le grade OMS\n"
        "- Mentionnez la confiance IA pour chaque resultat\n"
        "- Proposez des protocoles medicaux concretes (chirurgie, suivi, bilan)\n"
        "- Si confiance < 85%, suggerez un diagnostic differentiel\n"
        "- Repondez de maniere concise (3-6 phrases max)\n"
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
        self.model = None
        self.tokenizer = None
        from knowledge.icd11 import ICD11Client
        self.icd11 = ICD11Client(self.settings.icd11)
        self.model_loaded = False
        self.current_report: Optional[DiagnosticReport] = None
        self.conversation_history: list[dict[str, str]] = []
        self.icd11_codes: list[str] = []

    def load_model(self) -> None:
        """Load the Gemma 3 1B-IT model"""

        model_id = self.settings.model.llm_model_id
        logger.info(f"Loading {model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            use_fast=True,
            token=self.settings.model.hf_token
        )

        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
                token=self.settings.model.hf_token,
            )
            logger.info(f"Loaded {model_id} on GPU (float16) with eager attention (stability mode)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                token=self.settings.model.hf_token,
            )
            logger.warning(
                f"Loaded {model_id} on CPU. "
                "Expect slow inference."
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_loaded = True
        logger.info("Dialogue Agent model loaded")

    def set_report_context(self, report: DiagnosticReport) -> None:
        """Set the current diagnostic report as conversation context."""
        self.current_report = report
        self.conversation_history = []
        
        # Fetch ICD-11 codes if tumor detected
        self.icd11_codes = []
        if report.tumor_detected and self.icd11.available:
            query = f"{report.classification} brain tumor"
            results = self.icd11.search(query)
            for res in results[:5]:
                raw_id = res.get('id', '')
                segments = [s for s in raw_id.split('/') if s]
                entity_id = None
                for seg in reversed(segments):
                    if seg.replace('.', '').isdigit():
                        entity_id = seg
                        break
                if not entity_id:
                    continue
                code = self.icd11.get_mms_code(entity_id)
                if code:
                    self.icd11_codes.append(f"{code} ({res.get('title')})")
        
        logger.debug(f"Set report context for patient {report.patient_id}")

    def build_context_prompt(self) -> str:
        """Build a structured context string from the diagnostic report and RAG.

        Uses French labels for consistency with the system prompt.
        Strips the full clinical summary (already processed) to save tokens.
        """
        if self.current_report is None:
            return "Aucun rapport de diagnostic disponible."

        report = self.current_report
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
            if self.icd11_codes:
                parts.append(f"CIM-11 : {'; '.join(self.icd11_codes)}")
            # Include a short excerpt of the clinical summary (first 300 chars)
            if report.clinical_summary:
                excerpt = report.clinical_summary[:300]
                if len(report.clinical_summary) > 300:
                    excerpt += "..."
                parts.append(f"Resume clinique : {excerpt}")
        else:
            parts.append("Tumeur detectee : Non")
            parts.append("Examen normal.")

        # ── RAG Context (compact) ──
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
            rag_scores = []
            for q in queries:
                for chunk in self.rag_system.retrieve(q, top_k=2, min_score=0.3):
                    key = chunk.text[:80]
                    if key not in seen:
                        seen.add(key)
                        rag_parts.append(chunk.text)
                        rag_scores.append(getattr(chunk, "score"))

            if rag_parts:
                # Truncate RAG content to avoid overwhelming the 1B model
                rag_text = "\n---\n".join(rag_parts[:3])
                if len(rag_text) > 600:
                    rag_text = rag_text[:600] + "..."
                parts.append(f"\nConnaissances cliniques :\n{rag_text}")
                logger.debug(
                    f"[RAG] Dialogue: {len(rag_parts)} chunks "
                    f"(avg: {sum(rag_scores)/max(len(rag_scores),1):.3f})"
                )

        return "\n".join(parts)

    def build_messages(self, user_query: str) -> list[dict[str, str]]:
        """Build the chat messages list for the model's chat template.

        Structure:
        - system: short prompt + few-shot examples
        - context as first user/assistant pair (anchors the report data)
        - recent history (last 2 exchanges, not 4)
        - current query
        """
        context = self.build_context_prompt()

        # ── System message: prompt + few-shot examples ──
        examples_text = ""
        for ex in self.CHAT_EXAMPLES:
            examples_text += f"\nMedecin : {ex['user']}\nAssistant : {ex['assistant']}\n"

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Exemples de reponses attendues :\n"
            f"{examples_text}"
        )

        messages = [{"role": "system", "content": system_content}]

        # ── Inject report context as a structured user/assistant pair ──
        messages.append({
            "role": "user",
            "content": "Voici le rapport de diagnostic du patient. Memorisez-le.",
        })
        messages.append({
            "role": "assistant",
            "content": f"Rapport enregistre :\n{context}",
        })

        # ── Recent conversation history (last 2 exchanges) ──
        for exchange in self.conversation_history[-2:]:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})

        # ── Current query ──
        messages.append({"role": "user", "content": user_query})
        return messages

    @torch.no_grad()
    def chat(self, user_query: str) -> str:
        """
        Process a doctor's query and generate a response.

        Args:
            user_query: natural language question from the doctor

        Returns:
            Natural language response grounded in the diagnostic report
        """
        if not self.model_loaded:
            self.load_model()

        messages = self.build_messages(user_query)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )

        if torch.cuda.is_available() and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.settings.model.llm_max_new_tokens,
            temperature=self.settings.model.llm_temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Post-processing: strip leaked metadata lines
        response = re.sub(r'\[RAG Metrics:.*?\]', '', response).strip()
        response = re.sub(r'Requete RAG:.*?\n', '', response).strip()

        self.conversation_history.append({
            "user": user_query,
            "assistant": response,
        })

        return response

    @torch.no_grad()
    def chat_stream(self, user_query: str):
        """
        Process a doctor's query and generate a streamed response.

        Args:
            user_query: natural language question from the doctor

        Yields:
            Text chunks as they are generated by the model
        """
        if not self.model_loaded:
            self.load_model()

        messages = self.build_messages(user_query)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )

        if torch.cuda.is_available() and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.settings.model.llm_max_new_tokens,
            temperature=self.settings.model.llm_temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        full_response = ""
        for new_text in streamer:
            full_response += new_text
            yield new_text

        # Record the conversation history after stream completes
        cleaned = re.sub(r'\[RAG Metrics:.*?\]', '', full_response).strip()
        self.conversation_history.append({
            "user": user_query,
            "assistant": cleaned,
        })

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

    def get_confidence_explanation(self) -> str:
        """Explain the AI confidence scores and their clinical meaning."""
        return self.chat(
            "Expliquez le score de confiance de la classification tumorale. "
            "Que signifie-t-il cliniquement ? Est-il suffisant ?"
        )
