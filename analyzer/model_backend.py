"""
model_backend.py
-----------------

Este arquivo gerencia a IA principal do Analyzer.
Atualmente ele funciona em modo “placeholder”, mas já está preparado
para receber modelos reais:

- OpenAI API
- HuggingFace API
- LLaMA local via GGUF (llama-cpp-python)
- Groq API
- Modelos customizados

Quando você decidir qual IA usar, só ativamos a parte necessária.
"""

import os
import requests

class ModelBackend:

    
    def __init__(self, model_path=None, api_key=None, provider="placeholder"):
        """
        Inicializa o backend do modelo.

        model_path -> caminho para modelos locais (ex: LLaMA GGUF)
        api_key    -> chave de API se usar OpenAI, HuggingFace, Groq, etc.
        provider   -> tipo de IA usada: "placeholder", "openai", "local", "hf", etc.
        """

        self.model_path = model_path
        self.api_key = api_key
        self.provider = provider

        # DEBUG
        print(f"[ModelBackend] Inicializado com provider='{provider}', model_path={model_path}")

        # Placeholder seguro
        if provider == "placeholder":
            print("[ModelBackend] Modo placeholder ativado (IA simulada).")

        # Se um provider real for ativado, aqui você inicializa o cliente da IA.
        # Mas só faremos isso quando você escolher qual modelo quer usar.

    # ------------------------------------------------------------
    # Função principal: gerar resposta
    # ------------------------------------------------------------

    def initialize(self):
        """
        Inicializa o modelo real, caso necessário.
        No modo placeholder, apenas imprime que está pronto.
        Para IAs reais (OpenAI, LLaMA local, HuggingFace, etc),
        faremos o carregamento aqui.
        """
        print("[ModelBackend] Inicialização concluída (placeholder). Modelo pronto.")


    def generate_response(self, prompt, expert_mode=False):
        """
        Gera uma resposta usando o modelo configurado.
        Se provider == placeholder -> resposta simulada.
        """

        # -> MODO SIMULADO (funciona mesmo sem IA real)
        if self.provider == "placeholder":
            if expert_mode:
                return f"[Expert Mode] (simulado) Analisando tecnicamente: {prompt}"
            else:
                return f"(simulado) Resposta geral: {prompt}"

        # -> PROVIDERS REAIS (serão ativados quando definir a IA)
        if self.provider == "openai":
            return self._generate_openai(prompt, expert_mode)

        if self.provider == "local":
            return self._generate_local_llama(prompt, expert_mode)

        if self.provider == "hf":
            return self._generate_huggingface(prompt, expert_mode)

        # Se chegar aqui, provider inválido
        return "Erro: provider inválido."

    # ------------------------------------------------------------
    # Templates para IA real (implementaremos quando você escolher)
    # ------------------------------------------------------------

    def _generate_openai(self, prompt, expert_mode):
        return "OpenAI ainda não configurado."

    def _generate_local_llama(self, prompt, expert_mode):
        return "LLaMA local ainda não configurado."

    def _generate_huggingface(self, prompt, expert_mode):
        return "HuggingFace ainda não configurado."
