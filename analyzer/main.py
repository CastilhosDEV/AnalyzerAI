# analyzer/main.py
"""
Analyzer v5 - Main (robusto)
- GUI mantida (dark theme, thinking label separado)
- Worker threads + queue
- Usa ModelBackend (qwen2.5:7b-q4_K_M por padrão)
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import tkinter.font as tkfont
import threading
import queue
import time
import json
import os
import logging
from typing import Optional

from model_backend import ModelBackend

# Logging file for GUI
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "analyzer_gui.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
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

# Local settings file
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "analyzer_settings.json")

# Utility: run function in background thread
def run_in_thread(fn, daemon: bool = True) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=daemon)
    t.start()
    return t

# Settings helpers
def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"save_settings failed: {e}")

class AnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analyzer v5 — Avançado (Ryze)")
        self.root.configure(bg=BG)
        self.root.geometry("1100x760")
        self.root.minsize(900, 560)

        # load settings
        self.settings = load_settings()
        self.model_name = self.settings.get("model_name", "qwen2.5:7b-q4_K_M")
        self.host = self.settings.get("host", "http://127.0.0.1:11434")
        self.history_capacity = self.settings.get("history_capacity", 80)

        # backend setup
        self.backend = ModelBackend(provider="ollama", model_name=self.model_name, host=self.host, history_capacity=self.history_capacity, warmup=True)
        self.backend.initialize()

        # UI state
        self._q = queue.Queue()
        self._thinking = False
        self._thinking_job = None
        self._dots = 0
        self._last_user_prompt: Optional[str] = None
        self._last_response: Optional[str] = None

        # fonts
        self.base_font = tkfont.Font(family=FONT_NAME, size=FONT_SIZE)
        self.small_font = tkfont.Font(family=FONT_NAME, size=max(9, FONT_SIZE - 2))
        self.italic_font = tkfont.Font(family=FONT_NAME, size=FONT_SIZE, slant="italic")

        # setup UI
        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()

        # start monitor thread
        run_in_thread(self._monitor_loop)

        # initial greeting (backend may have done warmup)
        try:
            greeting = self.backend.initial_assistant_greeting()
        except Exception:
            greeting = "Olá, meu nome é Analyzer. Sou uma IA projetada para conversar naturalmente. O que deseja saber hoje?"
        self._append_chat("Analyzer", greeting, tag="analyzer")

        # start polling queue
        self.root.after(150, self._process_queue)
        logger.info("AnalyzerApp started")

    # ---------------- UI construction ----------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exportar conversa (TXT)", accelerator="Ctrl+S", command=self.on_export_txt)
        filemenu.add_command(label="Exportar histórico (JSON)", command=self.on_export_json)
        filemenu.add_separator()
        filemenu.add_command(label="Salvar configurações", command=self._save_settings)
        filemenu.add_separator()
        filemenu.add_command(label="Sair", command=self.root.quit)
        menubar.add_cascade(label="Arquivo", menu=filemenu)

        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Limpar chat (Ctrl+L)", command=self.on_clear)
        tools.add_command(label="Retry última (Ctrl+R)", command=self.on_retry)
        tools.add_separator()
        tools.add_command(label="Carregar histórico (JSON)...", command=self.on_load_history)
        menubar.add_cascade(label="Ferramentas", menu=tools)

        settings = tk.Menu(menubar, tearoff=0)
        settings.add_command(label="Configurar modelo/host...", command=self._open_settings_dialog)
        menubar.add_cascade(label="Configurações", menu=settings)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Sobre", command=self._show_about)
        menubar.add_cascade(label="Ajuda", menu=helpmenu)

        self.root.config(menu=menubar)

    def _build_ui(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        self.chat_display = scrolledtext.ScrolledText(main, wrap=tk.WORD, font=self.base_font, bg=CHAT_BG, fg=USER_COLOR, state=tk.DISABLED, padx=8, pady=8)
        self.chat_display.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        self.chat_display.tag_config("user", foreground=USER_COLOR, justify="right", font=self.base_font)
        self.chat_display.tag_config("analyzer", foreground=ANALYZER_COLOR, justify="left", font=self.base_font)
        self.chat_display.tag_config("meta", foreground=META_COLOR, justify="center", font=self.small_font)
        self.chat_display.tag_config("thinking", foreground="#888888", justify="left", font=self.italic_font)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0,12))

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(bottom, textvariable=self.input_var, font=self.base_font, bg="#1b1b1b", fg=USER_COLOR, insertbackground=USER_COLOR)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0,8))
        self.input_entry.bind("<Return>", lambda e: self.on_send())

        btn_send = tk.Button(bottom, text="Enviar (Ctrl+Enter)", command=self.on_send, bg="#233645", fg=USER_COLOR)
        btn_send.pack(side=tk.LEFT, padx=(0,6))

        btn_retry = tk.Button(bottom, text="Retry", command=self.on_retry, bg="#2e3b2e", fg=USER_COLOR)
        btn_retry.pack(side=tk.LEFT, padx=(0,6))

        btn_export = tk.Button(bottom, text="Exportar", command=self.on_export_txt, bg="#2e2e44", fg=USER_COLOR)
        btn_export.pack(side=tk.LEFT, padx=(0,6))

        btn_clear = tk.Button(bottom, text="Limpar", command=self.on_clear, bg="#3b2e2e", fg=USER_COLOR)
        btn_clear.pack(side=tk.LEFT, padx=(0,6))

        self.thinking_label = tk.Label(self.root, text="", bg=BG, fg=META_COLOR, font=self.small_font)
        self.thinking_label.pack(side=tk.BOTTOM, pady=(0,6))

        status_frame = tk.Frame(self.root, bg=BG)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Pronto")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, bg=BG, fg=META_COLOR, anchor="w", font=self.small_font)
        self.status_label.pack(fill=tk.X, padx=8, pady=4)

    # ---------------- UI helpers ----------------
    def _append_chat(self, who: str, text: str, tag: str = "analyzer"):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{ts}] {who}:\n"
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, header, "meta")
        self.chat_display.insert(tk.END, text + "\n\n", tag)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.yview(tk.END)
        logger.info(f"{who}: {text[:240]}")

    # ---------------- Actions ----------------
    def on_send(self):
        prompt = self.input_var.get().strip()
        if not prompt:
            return
        self._last_user_prompt = prompt
        self._append_chat("Você", prompt, tag="user")
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

    def on_export_txt(self):
        convo = self.chat_display.get("1.0", tk.END).strip()
        if not convo:
            messagebox.showinfo("Exportar", "Nada para exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(convo)
                messagebox.showinfo("Exportar", f"Conversa salva em:\n{path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao salvar: {e}")

    def on_export_json(self):
        data = self.backend.export_history()
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Exportar", f"Histórico salvo em:\n{path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao salvar JSON: {e}")

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

    def on_load_history(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json")])
        if not path:
            return
        try:
            self.backend.load_history_from_file(path)
            # refresh chat display
            self.chat_display.configure(state=tk.NORMAL)
            self.chat_display.delete("1.0", tk.END)
            for m in self.backend.export_history():
                role = m.get("role")
                content = m.get("content", "")
                if role == "system":
                    continue
                who = "Você" if role == "user" else "Analyzer"
                tag = "user" if role == "user" else "analyzer"
                self.chat_display.insert(tk.END, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {who}:\n", "meta")
                self.chat_display.insert(tk.END, content + "\n\n", tag)
            self.chat_display.configure(state=tk.DISABLED)
            messagebox.showinfo("Histórico", "Histórico carregado e exibido.")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar histórico: {e}")

    # ---------------- Worker/queue ----------------
    def _worker_generate(self, prompt: str):
        try:
            reply = self.backend.generate(prompt)
            self._last_response = reply
            # push into UI queue
            self._q.put(("message", reply))
        except Exception as e:
            logger.exception("Erro no worker_generate")
            self._q.put(("error", str(e)))
        finally:
            self._q.put(("status", "Pronto"))

    def _process_queue(self):
        try:
            while True:
                typ, payload = self._q.get_nowait()
                if typ == "status":
                    self.status_var.set(payload)
                    if payload == "Pronto":
                        self._stop_thinking()
                elif typ == "message":
                    self._append_chat("Analyzer", payload, tag="analyzer")
                elif typ == "error":
                    self._append_chat("Analyzer (erro)", payload, tag="meta")
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self._process_queue)

    # ---------------- Thinking (label only, not inserted in chat) ----------------
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
        # clear the label
        self.thinking_label.config(text="")
        # ensure status shows ready (may be updated by queue)
        self.status_var.set("Pronto")

    # ---------------- Monitor ----------------
    def _monitor_loop(self):
        try:
            import psutil
        except Exception:
            logger.info("psutil not installed; monitor disabled.")
            return
        while True:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory().percent
                if cpu >= 90 or mem >= 90:
                    self._q.put(("error", f"[Monitor] Alerta: CPU {cpu}%, RAM {mem}%"))
                    self._q.put(("status", "Alerta: sistema sobrecarregado"))
                time.sleep(6)
            except Exception as e:
                logger.exception(f"Monitor error: {e}")
                time.sleep(6)

    # ---------------- Settings dialog ----------------
    def _open_settings_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Configurações")
        dlg.geometry("440x260")
        dlg.configure(bg=BG)
        dlg.transient(self.root)

        tk.Label(dlg, text="Host Ollama:", bg=BG, fg=USER_COLOR).pack(anchor="w", padx=10, pady=(10,0))
        host_var = tk.StringVar(value=self.backend.host)
        host_entry = tk.Entry(dlg, textvariable=host_var, bg="#222", fg=USER_COLOR)
        host_entry.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(dlg, text="Modelo:", bg=BG, fg=USER_COLOR).pack(anchor="w", padx=10, pady=(8,0))
        model_var = tk.StringVar(value=self.backend.model_name)
        model_entry = tk.Entry(dlg, textvariable=model_var, bg="#222", fg=USER_COLOR)
        model_entry.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(dlg, text="History capacity:", bg=BG, fg=USER_COLOR).pack(anchor="w", padx=10, pady=(8,0))
        cap_var = tk.IntVar(value=self.backend.history_capacity)
        cap_spin = tk.Spinbox(dlg, from_=10, to=1000, textvariable=cap_var, bg="#222", fg=USER_COLOR)
        cap_spin.pack(fill=tk.X, padx=10, pady=4)

        def apply_changes():
            new_host = host_var.get().strip()
            new_model = model_var.get().strip()
            new_cap = int(cap_var.get())
            self.backend.host = new_host
            self.backend.model_name = new_model
            self.backend.history_capacity = new_cap
            self.backend._truncate_history()
            # save settings
            self.settings["host"] = new_host
            self.settings["model_name"] = new_model
            self.settings["history_capacity"] = new_cap
            save_settings(self.settings)
            messagebox.showinfo("Configurações", "Configuração aplicada. Reinicie o app se necessário.")
            dlg.destroy()

        tk.Button(dlg, text="Aplicar", command=apply_changes, bg="#2e3b2e", fg=USER_COLOR).pack(pady=10)

    # ---------------- Misc ----------------
    def _save_settings(self):
        save_settings(self.settings)
        messagebox.showinfo("Configurações", "Configurações salvas.")

    def _show_about(self):
        messagebox.showinfo("Sobre Analyzer", "Analyzer v5 — Criado por você. Ryze presente.")

    # ---------------- Keyboard shortcuts ----------------
    def _bind_shortcuts(self):
        self.root.bind_all("<Control-Return>", lambda e: self.on_send())
        self.root.bind_all("<Control-Return>", lambda e: self.on_send())
        self.root.bind_all("<Control-s>", lambda e: self.on_export_txt())
        self.root.bind_all("<Control-l>", lambda e: self.on_clear())
        self.root.bind_all("<Control-r>", lambda e: self.on_retry())

# ---------------- run ----------------
def main():
    root = tk.Tk()
    app = AnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
