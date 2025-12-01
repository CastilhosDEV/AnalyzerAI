# analyzer/model_backend.py
"""
ModelBackend (robusto) — adaptado para qwen2.5:7b-q4_K_M via Ollama local.

Principais mudanças aplicadas:
- model_name padrão: qwen2.5:7b-q4_K_M
- system prompt reforçado (não mencionar empresas, não inventar, obedecer apelidos)
- /api/chat preferido; fallback /api/generate
- streaming bufferizado (concatena internamente; retorna final)
- retries com exponential backoff (mais conservador)
- validação pós-processamento para remover menções indesejadas
- histórico estruturado e truncation (últimas N trocas)
"""

from typing import Optional, List, Dict
import requests
import json
import logging
import time
import os
import threading
import re

# Logger
logger = logging.getLogger("ModelBackend")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class ModelBackend:
    def __init__(
        self,
        provider: str = "ollama",
        model_name: Optional[str] = "qwen2.5:7b-q4_K_M",
        host: str = "http://127.0.0.1:11434",
        history_capacity: int = 80,
        warmup: bool = True,
        prefer_language: str = "pt-BR",
        max_retries: int = 3
    ):
        self.provider = provider
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.history_capacity = history_capacity
        self.prefer_language = prefer_language
        self.max_retries = max_retries

        # System prompt reforçado (curto, direto e seguro)
        self.system = (
            "Você é o Analyzer. Personalidade: técnica, direta, educada e obediente ao Mestre.\n"
            "Regras obrigatórias:\n"
            "- Responda preferencialmente em {lang} se o usuário usar {lang}.\n"
            "- Não mencione empresas, provedores ou detalhes de treinamento (não diga 'fui criado por X').\n"
            "- Não invente fatos; se incerto, peça clarificação.\n"
            "- Use linguagem natural, curta e objetiva. Trate o usuário conforme ele pedir (por ex. 'Criador', 'Mestre').\n"
            "- Não vaze raciocínios internos.\n"
            .format(lang=self.prefer_language)
        )

        # internal history
        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system}]

        # storage for history files
        self.storage_dir = os.path.join(os.path.dirname(__file__), "backend_storage")
        os.makedirs(self.storage_dir, exist_ok=True)

        # warmup flag
        self.warmup_done = False

        logger.info(f"[ModelBackend] init provider={self.provider} model={self.model_name} host={self.host}")

        # warmup in background so GUI isn't blocked
        if warmup and self.provider == "ollama":
            th = threading.Thread(target=self._warmup_safe, daemon=True)
            th.start()

    # -------------------- history helpers --------------------
    def _truncate_history(self) -> None:
        """Keep system at first position and keep last N user/assistant pairs."""
        if not self.history:
            self.history = [{"role": "system", "content": self.system}]
            return
        system = self.history[0] if self.history[0].get("role") == "system" else {"role": "system", "content": self.system}
        rest = [m for m in self.history[1:] if m.get("role") in ("user", "assistant")]
        if len(rest) > self.history_capacity:
            rest = rest[-self.history_capacity:]
        self.history = [system] + rest

    def push_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})
        self._truncate_history()

    def push_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._truncate_history()

    def export_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def clear_history(self, keep_system: bool = True) -> None:
        if keep_system:
            self.history = [{"role": "system", "content": self.system}]
        else:
            self.history = []

    def save_history_to_file(self, path: Optional[str] = None) -> str:
        if not path:
            ts = int(time.time())
            path = os.path.join(self.storage_dir, f"history_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        return path

    def load_history_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            if not loaded or loaded[0].get("role") != "system":
                loaded.insert(0, {"role": "system", "content": self.system})
            self.history = loaded
            self._truncate_history()

    # -------------------- warmup --------------------
    def initialize(self) -> None:
        """Ping host quickly (non-blocking usage expected)."""
        if self.provider != "ollama":
            logger.info("[ModelBackend] initialize: placeholder")
            return
        try:
            r = requests.get(self.host, timeout=2)
            logger.info(f"[ModelBackend] initialize ping {self.host} status {r.status_code}")
        except Exception as e:
            logger.warning(f"[ModelBackend] initialize failed: {e}")

    def _warmup_safe(self) -> None:
        try:
            self._warmup()
        except Exception as e:
            logger.warning(f"[ModelBackend] warmup error: {e}")

    def _warmup(self) -> None:
        if self.warmup_done:
            return
        logger.info("[ModelBackend] Warmup starting")
        try:
            _ = self._make_request(user_prompt="pong", prefer_chat=True, max_tokens=6, timeout=10)
            self.warmup_done = True
            logger.info("[ModelBackend] Warmup success")
        except Exception as e:
            logger.warning(f"[ModelBackend] Warmup failed: {e}")

    # -------------------- communication --------------------
    def _make_request(self, user_prompt: str, prefer_chat: bool = True, max_tokens: int = 1024, timeout: int = 90) -> str:
        """
        Central method. Tries /api/chat first with structured messages (stream=True).
        Reads streaming chunks and assembles response internally (buffered).
        """
        if self.provider != "ollama":
            return f"[SIMULADO] {user_prompt}"

        # prepare messages snapshot
        messages = list(self.history)
        messages.append({"role": "user", "content": user_prompt})

        attempt = 0
        while attempt <= self.max_retries:
            try:
                if prefer_chat:
                    return self._call_chat_api(messages, max_tokens=max_tokens, timeout=timeout)
                else:
                    return self._call_generate_api(messages, max_tokens=max_tokens, timeout=timeout)
            except requests.exceptions.RequestException as e:
                attempt += 1
                wait = min(2 * (2 ** (attempt - 1)), 20)  # faster backoff
                logger.warning(f"[ModelBackend] attempt {attempt}/{self.max_retries} failed: {e} - retry in {wait}s")
                time.sleep(wait)
            except Exception as e:
                logger.exception("[ModelBackend] unexpected error")
                raise
        raise requests.exceptions.RequestException("Max retries exceeded contacting Ollama")

    def _call_chat_api(self, messages: List[Dict[str, str]], max_tokens: int = 1024, timeout: int = 90) -> str:
        """
        Call Ollama /api/chat with streaming. Buffer fragments and return assembled text.
        This function is robust to multiple stream formats the Ollama may return.
        """
        url = f"{self.host}/api/chat"
        payload = {"model": self.model_name, "messages": messages, "stream": True, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}
        logger.debug("[ModelBackend] POST /api/chat")

        full_text = []
        try:
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                # read streaming lines
                for raw in resp.iter_lines(chunk_size=1, decode_unicode=False):
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except Exception:
                        # if not json, skip but keep working
                        continue

                    # multiple streaming formats supported
                    if isinstance(obj, dict):
                        # Ollama sometimes uses {"message": {"content": "..."}}
                        if "message" in obj and isinstance(obj["message"], dict):
                            c = obj["message"].get("content")
                            if isinstance(c, str) and c:
                                full_text.append(c)
                        elif "response" in obj and isinstance(obj["response"], str):
                            full_text.append(obj["response"])
                        elif "text" in obj and isinstance(obj["text"], str):
                            full_text.append(obj["text"])
                        elif "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                            c0 = obj["choices"][0]
                            if isinstance(c0, dict) and "text" in c0 and isinstance(c0["text"], str):
                                full_text.append(c0["text"])
                    # stop condition
                    if obj.get("done"):
                        break
        except requests.exceptions.RequestException:
            raise
        except Exception as e:
            logger.exception(f"[ModelBackend] unexpected error reading stream: {e}")
            raise

        assembled = "".join(full_text).strip()
        # post-process and sanitize
        assembled = self._post_process_reply(assembled)
        return assembled

    def _call_generate_api(self, messages: List[Dict[str, str]], max_tokens: int = 1024, timeout: int = 90) -> str:
        """
        Fallback to /api/generate (builds a single prompt).
        """
        url = f"{self.host}/api/generate"
        # craft prompt from recent history to preserve context
        system_part = self.system
        snippet = ""
        for m in messages[-(self.history_capacity + 1):]:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                continue
            if role == "user":
                snippet += f"Usuário: {content}\n"
            else:
                snippet += f"Analyzer: {content}\n"
        prompt = f"{system_part}\n\n{snippet}\nAnalyzer:"
        payload = {"model": self.model_name, "prompt": prompt, "stream": True, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}

        parts = []
        try:
            with requests.post(url, data=json.dumps(payload), headers=headers, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(chunk_size=1, decode_unicode=False):
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        if "response" in obj and isinstance(obj["response"], str):
                            parts.append(obj["response"])
                        elif "text" in obj and isinstance(obj["text"], str):
                            parts.append(obj["text"])
                        elif "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                            c0 = obj["choices"][0]
                            if isinstance(c0, dict) and "text" in c0:
                                parts.append(c0["text"])
                    if obj.get("done"):
                        break
        except requests.exceptions.RequestException:
            raise
        except Exception as e:
            logger.exception(f"[ModelBackend] generate fallback error: {e}")
            raise

        assembled = "".join(parts).strip()
        assembled = self._post_process_reply(assembled)
        return assembled

    # -------------------- post processing & validation --------------------
    def _post_process_reply(self, text: str) -> str:
        """Sanitize and enforce system rules on reply."""
        if not text:
            return text

        # basic sanitization: trim repeated whitespace
        text = re.sub(r"\s+\n", "\n", text).strip()

        # ban mentions of specific provider names or training statements
        forbidden = [
            r"anthropic", r"openai", r"meta", r"trained by", r"trained on", r"i was trained", r"i am a large"
        ]
        lowered = text.lower()
        for f in forbidden:
            if re.search(f, lowered):
                # remove offending sentence(s)
                text = re.sub(r"[^.?!]*(" + f + r")[^.?!]*[.?!]?", "", text, flags=re.IGNORECASE).strip()

        # if reply becomes empty after sanitization, give a safe fallback
        if not text:
            text = "[ERRO] Resposta removida por política interna. Peça para reformular a pergunta."

        # coherence check (simple heuristic): if response is extremely short, ask to clarify
        if len(text) < 8:
            text = text + "\n\nSe precisar, descreva com mais detalhes sua pergunta."

        return text

    # -------------------- public interface --------------------
    def generate(self, user_prompt: str, max_tokens: int = 1024, prefer_chat: bool = True) -> str:
        user_prompt = user_prompt.strip()
        if not user_prompt:
            return ""

        # push and truncate history
        self.push_user(user_prompt)

        if self.provider != "ollama":
            reply = f"[SIMULADO] {user_prompt}"
            self.push_assistant(reply)
            return reply

        try:
            reply = self._make_request(user_prompt, prefer_chat=prefer_chat, max_tokens=max_tokens, timeout=90)
            if not reply:
                reply = "[ERRO] O modelo retornou resposta vazia."
            self.push_assistant(reply)
            return reply
        except Exception as e:
            logger.exception("[ModelBackend] generate failed")
            fallback = f"[ERRO OLLAMA] {e}. Resposta fallback."
            self.push_assistant(fallback)
            return fallback

    def initial_assistant_greeting(self) -> str:
        greeting = (
            "Olá, meu nome é Analyzer. Sou uma IA projetada para conversar naturalmente, manter contexto "
            "e ajudar com tarefas técnicas e gerais. O que deseja saber hoje?"
        )
        self.push_assistant(greeting)
        return greeting
