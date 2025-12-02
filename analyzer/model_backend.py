# analyzer/model_backend.py
"""
ModelBackend (avançado, versão 'mestre'):
- Usa Ollama local (modelo qwen2.5:7b-q4_K_M por padrão)
- /api/chat preferido (stream=True) com leitura buffered e montagem final no backend
- fallback para /api/generate
- warmup não-bloqueante
- history role-structured (system/user/assistant)
- local solver para aritmética simples (reduz latência em perguntas matemáticas)
- retries com exponential backoff
- post-process para remover menções proibidas (providers/trainers)
"""

from typing import Optional, List, Dict, Any, Tuple
import requests
import json
import logging
import os
import time
import threading
import re
import ast

# logger
logger = logging.getLogger("ModelBackend")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# load settings
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "analyzer_settings.json")
if not os.path.exists(SETTINGS_PATH):
    raise FileNotFoundError(f"Settings file not found: {SETTINGS_PATH}")

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# defaults from settings
DEFAULT_MODEL = SETTINGS.get("model_name", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_HOST = SETTINGS.get("host", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT = SETTINGS.get("network", {}).get("ollama_timeout_seconds", 90)
OLLAMA_MAX_RETRIES = SETTINGS.get("network", {}).get("ollama_max_retries", 3)
SYSTEM_PROMPT = SETTINGS.get("system_prompt", "Você é o Analyzer — assistente técnico, direto e obediente ao Mestre.")
PREFERRED_LANGUAGE = "pt-BR"

# helper: safe arithmetic evaluator (only arithmetic expressions)
def safe_eval_arith(expr: str):
    """
    Safely evaluate a mathematical expression limited to arithmetic ops.
    Prevents arbitrary code execution by validating AST nodes.
    """
    expr = expr.replace("^", "**")
    try:
        node = ast.parse(expr, mode="eval")
    except Exception as e:
        raise ValueError("invalid expression") from e
    for n in ast.walk(node):
        if isinstance(n, (ast.Call, ast.Name, ast.Attribute, ast.Subscript, ast.Lambda, ast.Import, ast.ImportFrom)):
            raise ValueError("unsafe expression")
    compiled = compile(node, "<ast>", "eval")
    return eval(compiled, {"__builtins__": None}, {})

# basic enunciado solver (linear forms like "se eu dobro um número e somo 6 o resultado é 26")
def solve_simple_enunciado(text: str) -> Optional[str]:
    s = text.lower()
    # try pattern: "dobro ... somo 6 ... resultado é 26"
    m = re.search(r"dobr[oa] (?:um )?n[úu]mero.*som[ao]\s*(-?\d+\.?\d*)[, ]+.*resultado.*?(-?\d+\.?\d*)", s)
    if m:
        try:
            mult = 2.0
            add = float(m.group(1))
            result = float(m.group(2))
            original = (result - add) / mult
            if abs(round(original) - original) < 1e-9:
                original = int(round(original))
            return f"O número original é {original}."
        except Exception:
            pass
    # broader heuristic: find "dobro" or "triplo" or "multiplica por N" and addition/subtraction then result
    mult = None
    if "dobr" in s:
        mult = 2.0
    elif "trip" in s:
        mult = 3.0
    else:
        m_mul = re.search(r"multiplic[ao]r? (?:por )?(-?\d+\.?\d*)", s)
        if m_mul:
            mult = float(m_mul.group(1))
    m_res = re.search(r"resultado.*?(-?\d+\.?\d*)", s)
    m_add = re.search(r"(som[ao]|mais)\s*(-?\d+\.?\d*)", s)
    sign = 0.0
    if m_add:
        sign = float(m_add.group(2))
    if mult and m_res:
        try:
            result = float(m_res.group(1))
            original = (result - sign) / mult
            if abs(round(original) - original) < 1e-9:
                original = int(round(original))
            return f"O número original é {original}."
        except Exception:
            pass
    return None

class ModelBackend:
    def __init__(self, provider: str = "ollama", model_name: Optional[str] = None, host: str = None):
        self.provider = provider
        self.model_name = model_name or DEFAULT_MODEL
        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.history: List[Dict[str,str]] = [{"role":"system","content":SYSTEM_PROMPT}]
        self.warmup_done = False
        self.lock = threading.Lock()
        logger.info(f"[ModelBackend] init provider={self.provider} model={self.model_name} host={self.host}")
        # warmup in background
        t = threading.Thread(target=self._warmup_safe, daemon=True)
        t.start()

    # ---------------- history helpers ----------------
    def push_user(self, text: str):
        with self.lock:
            self.history.append({"role":"user","content":text})
            # keep last N messages
            max_msgs = SETTINGS.get("max_context_messages", 120)
            if len(self.history) > max_msgs + 1:  # +1 for system
                # keep system + last max_msgs
                self.history = [self.history[0]] + self.history[-max_msgs:]

    def push_assistant(self, text: str):
        with self.lock:
            self.history.append({"role":"assistant","content":text})
            max_msgs = SETTINGS.get("max_context_messages", 120)
            if len(self.history) > max_msgs + 1:
                self.history = [self.history[0]] + self.history[-max_msgs:]

    def export_history(self) -> List[Dict[str,str]]:
        with self.lock:
            return list(self.history)

    def clear_history(self, keep_system: bool = True):
        with self.lock:
            if keep_system:
                self.history = [{"role":"system","content":SYSTEM_PROMPT}]
            else:
                self.history = []

    # ---------------- warmup ----------------
    def _warmup_safe(self):
        try:
            self._warmup()
        except Exception as e:
            logger.info(f"[ModelBackend] warmup exception: {e}")

    def _warmup(self):
        if self.warmup_done:
            return
        try:
            # tiny request to reduce first-response oddities
            _ = self.generate("pong", prefer_chat=True, max_tokens=8, timeout=8)
            self.warmup_done = True
            logger.info("[ModelBackend] warmup completed")
        except Exception as e:
            logger.warning(f"[ModelBackend] warmup failed: {e}")

    # ---------------- networking helpers ----------------
    def _compose_messages(self, user_prompt: str) -> List[Dict[str,str]]:
        with self.lock:
            msgs = list(self.history)
        msgs.append({"role":"user","content":user_prompt})
        return msgs

    def _call_chat_api(self, messages: List[Dict[str,str]], max_tokens: int = 1024, timeout: int = OLLAMA_TIMEOUT) -> str:
        url = f"{self.host}/api/chat"
        payload = {"model": self.model_name, "messages": messages, "stream": True, "max_tokens": max_tokens}
        headers = {"Content-Type":"application/json"}
        parts: List[str] = []
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(obj, dict):
                    # support multiple formats
                    if "message" in obj and isinstance(obj["message"], dict):
                        content = obj["message"].get("content")
                        if isinstance(content, str):
                            parts.append(content)
                    elif "response" in obj and isinstance(obj["response"], str):
                        parts.append(obj["response"])
                    elif "text" in obj and isinstance(obj["text"], str):
                        parts.append(obj["text"])
                    elif "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                        c0 = obj["choices"][0]
                        if isinstance(c0, dict) and "text" in c0:
                            parts.append(c0["text"])
                if obj.get("done"):
                    break
        assembled = "".join(parts).strip()
        return self._post_process_reply(assembled)

    def _call_generate_api(self, messages: List[Dict[str,str]], max_tokens:int=1024, timeout:int=OLLAMA_TIMEOUT) -> str:
        url = f"{self.host}/api/generate"
        # fallback: build prompt from system + recent history
        system_part = SYSTEM_PROMPT
        snippet = ""
        for m in messages[-(SETTINGS.get("max_context_messages",120)+1):]:
            role = m.get("role")
            content = m.get("content","")
            if role == "system":
                continue
            if role == "user":
                snippet += f"Usuário: {content}\n"
            else:
                snippet += f"Analyzer: {content}\n"
        prompt = f"{system_part}\n\n{snippet}\nAnalyzer:"
        payload = {"model": self.model_name, "prompt": prompt, "stream": True, "max_tokens": max_tokens}
        headers = {"Content-Type":"application/json"}
        parts = []
        with requests.post(url, data=json.dumps(payload), headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=False):
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
        assembled = "".join(parts).strip()
        return self._post_process_reply(assembled)

    # ---------------- post process ----------------
    def _post_process_reply(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        # remove forbidden provider mentions
        forbidden = [r"anthropic", r"openai", r"meta", r"trained by", r"i was trained", r"i am a"]
        low = text.lower()
        for f in forbidden:
            if re.search(f, low):
                text = re.sub(r"(?i)[^.?!]*" + re.escape(f) + r"[^.?!]*[.?!]?", "", text).strip()
        if not text:
            return "[ERRO] Resposta removida por política interna."
        return text

    # ---------------- main public interface ----------------
    def generate(self, user_prompt: str, max_tokens: int = 1024, prefer_chat: bool = True, timeout: int = OLLAMA_TIMEOUT) -> str:
        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            return ""
        # push user to history
        try:
            self.push_user(user_prompt)
        except Exception:
            pass

        # local solver for speed on math/simple enunciados
        try:
            # direct arithmetic: "qual é 2+2" or "quanto é 2+2"
            m_ar = re.search(r"(qual(?: é| e)?|quanto(?: é| )?)\s*([0-9\.\+\-\*\/\^\(\) ]+)$", user_prompt.lower())
            if m_ar:
                expr = m_ar.group(2).strip()
                val = safe_eval_arith(expr)
                ans = str(val)
                self.push_assistant(ans)
                return ans
        except Exception:
            pass
        # enunciado solver
        try:
            s = solve_simple_enunciado(user_prompt)
            if s:
                self.push_assistant(s)
                return s
        except Exception:
            pass

        # network / model path
        if self.provider != "ollama":
            reply = f"[SIMULADO] {user_prompt}"
            try:
                self.push_assistant(reply)
            except Exception:
                pass
            return reply

        attempt = 0
        last_exc = None
        while attempt <= OLLAMA_MAX_RETRIES:
            try:
                messages = self._compose_messages(user_prompt)
                if prefer_chat:
                    reply = self._call_chat_api(messages, max_tokens=max_tokens, timeout=timeout)
                else:
                    reply = self._call_generate_api(messages, max_tokens=max_tokens, timeout=timeout)
                if not reply:
                    reply = "[ERRO] Resposta vazia do modelo."
                try:
                    self.push_assistant(reply)
                except Exception:
                    pass
                return reply
            except requests.exceptions.RequestException as e:
                last_exc = e
                attempt += 1
                wait = min(2 * (2 ** (attempt - 1)), 30)
                logger.warning(f"[ModelBackend] request error attempt {attempt}: {e} -> waiting {wait}s")
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                logger.exception("[ModelBackend] unexpected error")
                break

        fallback = (
            "[ERRO OLLAMA] Max retries exceeded contacting Ollama. Resposta fallback.\n\n"
            "Verifique se Ollama está rodando e se o modelo foi baixado. "
            f"Host: {self.host} | Modelo: {self.model_name}\n"
            "Sugestão: rode `ollama status` e `ollama list` no terminal."
        )
        try:
            self.push_assistant(fallback)
        except Exception:
            pass
        return fallback

    def initial_assistant_greeting(self) -> str:
        greeting = (
            "Olá, meu nome é Analyzer. Sou uma IA projetada para conversar naturalmente, manter contexto "
            "e ajudar com tarefas técnicas e gerais. O que deseja saber hoje?"
        )
        try:
            self.push_assistant(greeting)
        except Exception:
            pass
        return greeting

# end of model_backend.py
