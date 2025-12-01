# analyzer/utils.py
import threading

def run_in_thread(fn, daemon: bool = True):
    """
    Executa função fn em thread separada e inicia imediatamente.
    Retorna objeto Thread.
    """
    t = threading.Thread(target=fn, daemon=daemon)
    t.start()
    return t
