# analyzer/monitor.py
"""
Monitor proativo simples (psutil).
Imprime avisos no console; roda em thread separada.
"""

import psutil
import time
import logging

logging.getLogger().setLevel(logging.INFO)

class SystemMonitor:
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._running = True

    def start_monitoring(self):
        logging.info("[Monitor] iniciando")
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory().percent
                disk = psutil.disk_usage("/").percent

                if cpu >= 90:
                    logging.warning(f"[Monitor] CPU alto: {cpu}%")
                if mem >= 85:
                    logging.warning(f"[Monitor] RAM alta: {mem}%")
                if disk >= 90:
                    logging.warning(f"[Monitor] Disco cheio: {disk}%")
            except Exception as e:
                logging.exception(f"[Monitor] erro: {e}")
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        logging.info("[Monitor] parando")
