# analyzer/main.py
"""
Analyzer v5 - Main (Avançado, integrado)
- Mantém GUI e aparência (dark)
- Usa ModelBackend (Ollama qwen2.5 quantizado)
- Integra SystemMonitor via queue
- Thinking label separado da conversa (não insere "pensando" no chat)
- Export/Import, Retry, Clean, Settings dialog integrated
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import tkinter.font as tkfont
import queue
import threading
import time
import json
import os
import logging
from typing import Optional

# imports for backend + monitor
from model_backend import ModelBackend
from monitor import SystemMonitor

# logging file
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR,"analyzer_gui.log"), level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AnalyzerGUI")
logger.setLevel(logging.INFO)

# Visual constants
BG = "#0f1113"
CHAT_BG = "#0b0c0d"
USER_COLOR = "#FFFFFF"
ANALYZER_COLOR = "#7CFF6B"
META_COLOR = "#9AA5B1"
FONT_NAME = "Consolas"
FONT_SIZE = 11

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "analyzer_setting.json")

# util to load settings
def load_settings() -> dict:
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings(data: dict):
    try:
        with open(SETTINGS_PATH,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"save_settings failed: {e}")

# thread helper
def run_in_thread(fn, daemon=True):
    t = threading.Thread(target=fn, daemon=daemon)
    t.start()
    return t

class AnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyzer v5 — Avançado")
        self.root.configure(bg=BG)
        self.root.geometry("1100x720")
        self.root.minsize(780,480)

        # settings
        self.settings = load_settings()
        # create default settings if missing
        if not self.settings:
            try:
                with open(SETTINGS_PATH,"r",encoding="utf-8") as f:
                    self.settings = json.load(f)
            except Exception:
                self.settings = {}

        self._q = queue.Queue()
        # backend
        model_name = self.settings.get("model_name")
        host = self.settings.get("host")
        self.backend = ModelBackend(provider="ollama", model_name=model_name, host=host)
        self.backend.initialize = getattr(self.backend, "initialize", lambda: None)
        # monitor (pass queue so monitor can emit events)
        self.monitor = SystemMonitor(queue=self._q, interval=self.settings.get("diagnostic_mode", {}).get("auto_scan_interval", 8))
        run_in_thread(self.monitor.start_monitoring)

        # UI state
        self._thinking = False
        self._thinking_job = None
        self._dots = 0
        self._last_user_prompt: Optional[str] = None
        self._last_response: Optional[str] = None

        # fonts
        self.base_font = tkfont.Font(family=FONT_NAME, size=FONT_SIZE)
        self.small_font = tkfont.Font(family=FONT_NAME, size=max(9, FONT_SIZE-2))
        self.italic_font = tkfont.Font(family=FONT_NAME, size=FONT_SIZE, slant="italic")

        self._build_ui()
        # initial greeting from backend
        try:
            g = self.backend.initial_assistant_greeting()
        except Exception:
            g = "Olá, meu nome é Analyzer. Sou uma IA projetada para conversar naturalmente. O que deseja saber hoje?"
        self._append_chat("Analyzer", g, "analyzer")

        # poll queue
        self.root.after(150, self._process_queue)
        logger.info("AnalyzerApp initialized")

    def _build_ui(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self.chat_display = scrolledtext.ScrolledText(top, wrap=tk.WORD, bg=CHAT_BG, fg=USER_COLOR, font=self.base_font, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # tags
        self.chat_display.tag_config("user", foreground=USER_COLOR, justify="right", font=self.base_font)
        self.chat_display.tag_config("analyzer", foreground=ANALYZER_COLOR, justify="left", font=self.base_font)
        self.chat_display.tag_config("meta", foreground=META_COLOR, justify="center", font=self.small_font)
        # thinking tag defined but we won't insert thinking into chat
        try:
            self.chat_display.tag_config("thinking", foreground="#888888", justify="left", font=self.italic_font)
        except Exception:
            # older tkinter may reject font tuple; ignore
            pass

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill=tk.X, padx=12, pady=(0,12))

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(bottom, textvariable=self.input_var, font=self.base_font, bg="#1b1b1b", fg=USER_COLOR, insertbackground=USER_COLOR)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0,8))
        self.input_entry.bind("<Return>", lambda e: self.on_send())

        send_btn = tk.Button(bottom, text="Enviar", command=self.on_send, bg="#233645", fg=USER_COLOR)
        send_btn.pack(side=tk.LEFT, padx=(0,6))

        retry_btn = tk.Button(bottom, text="Retry", command=self.on_retry, bg="#2e3b2e", fg=USER_COLOR)
        retry_btn.pack(side=tk.LEFT, padx=(0,6))

        export_btn = tk.Button(bottom, text="Exportar", command=self.on_export, bg="#2e2e44", fg=USER_COLOR)
        export_btn.pack(side=tk.LEFT, padx=(0,6))

        clear_btn = tk.Button(bottom, text="Limpar", command=self.on_clear, bg="#3b2e2e", fg=USER_COLOR)
        clear_btn.pack(side=tk.LEFT, padx=(0,6))

        self.thinking_label = tk.Label(self.root, text="", bg=BG, fg=META_COLOR, font=self.small_font)
        self.thinking_label.pack(pady=(4,0))

        self.status_var = tk.StringVar(value="Pronto")
        status = tk.Label(self.root, textvariable=self.status_var, bg=BG, fg=META_COLOR, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _append_chat(self, who: str, text: str, tag: str):
        self.chat_display.configure(state=tk.NORMAL)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{ts}] {who}:\n"
        self.chat_display.insert(tk.END, header, "meta")
        self.chat_display.insert(tk.END, text + "\n\n", tag)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.yview(tk.END)
        logger.info(f"{who}: {text[:300]}")

    # actions
    def on_send(self):
        prompt = self.input_var.get().strip()
        if not prompt:
            return
        self._last_user_prompt = prompt
        self._append_chat("Você", prompt, "user")
        self.input_var.set("")
        self.status_var.set("Gerando...")
        self._start_thinking()
        run_in_thread(lambda: self._worker_generate(prompt))

    def on_retry(self):
        if not self._last_user_prompt:
            messagebox.showinfo("Retry", "Não há mensagem anterior para reenviar.")
            return
        self.input_var.set(self._last_user_prompt)
        self.on_send()

    def on_export(self):
        text = self.chat_display.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Exportar", "Nada para exportar.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
        if filename:
            with open(filename,"w",encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("Exportar", f"Conversa salva em:\n{filename}")

    def on_clear(self):
        if not messagebox.askyesno("Limpar", "Deseja limpar todo o histórico de chat?"):
            return
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.backend.clear_history(keep_system=True)
        self._last_user_prompt = None
        self._last_response = None
        self.status_var.set("Pronto")

    # worker
    def _worker_generate(self, prompt: str):
        try:
            resp = self.backend.generate(prompt)
            self._last_response = resp
            self._q.put(("message", resp))
        except Exception as e:
            logger.exception("worker_generate error")
            self._q.put(("error", str(e)))
        finally:
            self._q.put(("status", "Pronto"))

    # queue processing (including monitor events)
    def _process_queue(self):
        try:
            while True:
                typ, payload = self._q.get_nowait()
                if typ == "status":
                    self.status_var.set(payload)
                    if payload == "Pronto":
                        self._stop_thinking()
                elif typ == "message":
                    self._append_chat("Analyzer", payload, "analyzer")
                elif typ == "error":
                    self._append_chat("Analyzer (erro)", payload, "meta")
                elif typ == "monitor_alert":
                    # payload contains cpu/mem/disk/top_procs
                    detail = payload
                    summary = f"Alerta: CPU {detail['cpu']}% | RAM {detail['mem']}% | DISCO {detail['disk']}%\nTop processos:\n"
                    for p in detail.get("top_procs", [])[:3]:
                        summary += f"PID {p.get('pid')} - {p.get('name')} CPU% {p.get('cpu_percent')} MEM% {p.get('memory_percent')}\n"
                    # show in chat and also popup
                    self._append_chat("Analyzer (Monitor)", summary, "meta")
                    # if autonomy is not B3 ask user to act (monitor may auto-act depending on settings)
                    settings = load_settings()
                    autonomy = settings.get("diagnostic_mode", {}).get("autonomy", "B2")
                    if autonomy != "B3":
                        def ask_handle():
                            ans = messagebox.askyesno("Monitor", summary + "\nDeseja encerrar o processo mais pesado?")
                            if ans:
                                top = detail.get("top_procs",[])
                                if top:
                                    pid = top[0].get("pid")
                                    ok, msg = SystemMonitor.kill_process(self.monitor, pid) if hasattr(SystemMonitor, "kill_process") else (False,"kill fn not available")
                                    self._append_chat("Analyzer (Monitor)", f"Ação: {msg}", "meta")
                        self.root.after(50, ask_handle)
                elif typ == "monitor_suggest":
                    detail = payload
                    suggestion = f"Sugestão: {detail.get('suggest')} (disk {detail.get('disk')}%)"
                    self._append_chat("Analyzer (Monitor)", suggestion, "meta")
                    # ask user for cleaning
                    def ask_clean():
                        ans = messagebox.askyesno("Monitor - Sugestão", suggestion + "\nDeseja limpar temporários?")
                        if ans:
                            res = SystemMonitor.clean_temp_folder(None)
                            removed = len(res.get("removed", []))
                            failed = len(res.get("failed", []))
                            self._append_chat("Analyzer (Monitor)", f"Limpeza concluída. Removidos: {removed}, Falhas: {failed}", "meta")
                    self.root.after(50, ask_clean)
                elif typ == "monitor_action":
                    act = payload
                    self._append_chat("Analyzer (Monitor)", f"Ação executada: {act}", "meta")
                elif typ == "monitor_log":
                    msg = payload.get("msg") if isinstance(payload, dict) else str(payload)
                    self._append_chat("Analyzer (Monitor)", msg, "meta")
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._process_queue)

    # thinking animation (label only)
    def _start_thinking(self):
        if self._thinking:
            return
        self._thinking = True
        self._dots = 0
        self._animate_thinking()

    def _animate_thinking(self):
        if not self._thinking:
            self.thinking_label.config(text="")
            return
        dots = "." * (self._dots % 4)
        self.thinking_label.config(text=f"Analyzer está pensando{dots}")
        self._dots += 1
        self._thinking_job = self.root.after(450, self._animate_thinking)

    def _stop_thinking(self):
        if not self._thinking:
            return
        self._thinking = False
        if self._thinking_job:
            try:
                self.root.after_cancel(self._thinking_job)
            except Exception:
                pass
            self._thinking_job = None
        self.thinking_label.config(text="")
        self.status_var.set("Pronto")

# run
def main():
    root = tk.Tk()
    app = AnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
