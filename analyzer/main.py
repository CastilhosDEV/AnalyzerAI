# analyzer/main.py
"""
Analyzer - GUI + Monitor + LLM backend (esqueleto)

Características:
 - Janela Tkinter com área de chat (entrada + histórico)
 - Botão "Expert Mode" para alternar comportamento do modelo
 - Monitor proativo (psutil) que notifica GUI/usuário
 - Comunicação com ModelBackend (abstração)
 - Execução de chamadas longas em threads para não travar a GUI
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
from model_backend import ModelBackend
from monitor import SystemMonitor
from utils import run_in_thread
import queue
import time

class AnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Analyzer — Ryze")
        self.root.geometry("800x600")

        # Modelo backend (placeholder)
        self.backend = ModelBackend(model_path=None)
        self.backend.initialize()

        # Estado
        self.expert_mode = False
        self.response_queue = queue.Queue()

        # Criar UI
        self._build_ui()

        # Monitor proativo
        self.monitor = SystemMonitor(interval=5.0)
        self.monitor.on_alert = self.handle_system_alert
        self.monitor.start()

        # Poll para processar respostas geradas em threads
        self.root.after(200, self._poll_response_queue)

    def _build_ui(self):
        # Frame principal
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Histórico de mensagens (scroll)
        self.chat_display = scrolledtext.ScrolledText(top_frame, state='disabled', wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Frame inferior: entrada + botões
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=8, pady=6)

        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(bottom_frame, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
        self.input_entry.bind("<Return>", lambda e: self.on_send())

        send_btn = tk.Button(bottom_frame, text="Enviar", command=self.on_send)
        send_btn.pack(side=tk.LEFT)

        # Toggle Expert Mode
        self.expert_btn = tk.Button(bottom_frame, text="Ativar Expert Mode", command=self.toggle_expert)
        self.expert_btn.pack(side=tk.LEFT, padx=(6,0))

        # Status bar
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = tk.Label(self.root, textvariable=self.status_var, anchor='w')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_expert(self):
        self.expert_mode = not self.expert_mode
        self.expert_btn.config(text=("Desativar Expert Mode" if self.expert_mode else "Ativar Expert Mode"))
        self._append_chat("Sistema", f"Expert Mode {'ativado' if self.expert_mode else 'desativado'}")

    def _append_chat(self, who: str, text: str):
        """
        Escreve no widget de chat de forma thread-safe (chamado apenas na thread da GUI).
        """
        self.chat_display.configure(state='normal')
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.chat_display.insert(tk.END, f"[{timestamp}] {who}: {text}\n\n")
        self.chat_display.configure(state='disabled')
        # rolar para o final
        self.chat_display.yview(tk.END)

    def on_send(self):
        prompt = self.input_var.get().strip()
        if not prompt:
            return
        # mostra prompt no chat
        self._append_chat("Você", prompt)
        self.input_var.set("")
        # chama geração em thread para não travar GUI
        run_in_thread(lambda: self._generate_async(prompt))

    def _generate_async(self, prompt: str):
        try:
            # indica que estamos gerando
            self.response_queue.put(("status", "Gerando resposta..."))
            # chama backend.generate — aqui pode demorar dependendo do backend
            response = self.backend.generate(prompt, expert_mode=self.expert_mode)
            # coloca resultado na fila para ser processado na thread principal
            self.response_queue.put(("message", response))
        except Exception as e:
            self.response_queue.put(("error", str(e)))
        finally:
            self.response_queue.put(("status", "Pronto"))

    def _poll_response_queue(self):
        """
        Executado na thread da GUI periodicamente para processar mensagens vindas das threads de worker.
        """
        try:
            while True:
                typ, payload = self.response_queue.get_nowait()
                if typ == "status":
                    self.status_var.set(payload)
                elif typ == "message":
                    self._append_chat("Analyzer", payload)
                elif typ == "error":
                    self._append_chat("Erro", payload)
                    messagebox.showerror("Erro na geração", payload)
        except queue.Empty:
            pass
        finally:
            # reagenda
            self.root.after(200, self._poll_response_queue)

    def handle_system_alert(self, issue_type: str, details: dict):
        """
        Callback do monitor de sistema. Aqui decidimos como notificar o usuário.
        IMPORTANTE: chamadas do monitor acontecem em thread separada ->
        usar `root.after` para interagir com GUI com segurança.
        """
        def notify():
            # Mensagem resumida para o chat
            short = f"[ALERTA] {issue_type}: {details}"
            self._append_chat("Monitor", short)
            # Pop-up para problemas críticos (opcional)
            if issue_type in ("cpu_high", "ram_high", "disk_high"):
                # perguntar ao usuário se quer ver recomendações
                if messagebox.askyesno("Alerta do Sistema", f"{issue_type} detectado. Ver recomendações?"):
                    # enviar pergunta ao modelo em Expert Mode pedindo solução
                    self._append_chat("Sistema", "Gerando recomendações (Expert Mode) ...")
                    # chamada assíncrona: forçamos expert_mode True temporariamente
                    run_in_thread(lambda: self._generate_async(f"Recomendações para {issue_type}: {details}"))
        # garantir execução na thread da GUI
        self.root.after(0, notify)

def main():
    root = tk.Tk()
    app = AnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
