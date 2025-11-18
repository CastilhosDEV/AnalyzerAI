# analyzer/monitor.py
import threading
import time
import psutil
from typing import Callable, Optional

class SystemMonitor(threading.Thread):
    """
    Thread que monitora o sistema periodicamente e chama callbacks
    quando encontra condições que merecem atenção.
    """

    def __init__(self, interval: float = 5.0):
        super().__init__(daemon=True)  # daemon -> não bloqueia encerramento do programa
        self.interval = interval
        self._stop_event = threading.Event()
        # callbacks: função que recebe (issue_type:str, details:dict)
        self.on_alert: Optional[Callable[[str, dict], None]] = None

        # thresholds (ajuste conforme desejar)
        self.cpu_threshold = 90.0     # %
        self.ram_threshold = 85.0     # %
        self.disk_threshold = 90.0    # %
        self.swap_threshold = 50.0    # %
        self.high_threads_threshold = 300

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self.check_once()
            time.sleep(self.interval)

    def check_once(self):
        # CPU
        cpu = psutil.cpu_percent(interval=None)
        if cpu >= self.cpu_threshold:
            self._alert("cpu_high", {"cpu_percent": cpu})

        # RAM
        vm = psutil.virtual_memory()
        ram_percent = vm.percent
        if ram_percent >= self.ram_threshold:
            self._alert("ram_high", {"ram_percent": ram_percent, "available": vm.available})

        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        if disk.percent >= self.disk_threshold:
            self._alert("disk_high", {"disk_percent": disk.percent, "total": disk.total, "free": disk.free})

        # Swap
        swap = psutil.swap_memory()
        if swap.percent >= self.swap_threshold:
            self._alert("swap_high", {"swap_percent": swap.percent})

        # Many processes/threads (heurístico)
        total_threads = sum(p.num_threads() for p in psutil.process_iter(attrs=[], ad_value=0))
        if total_threads >= self.high_threads_threshold:
            self._alert("many_threads", {"total_threads": total_threads})

    def _alert(self, issue_type: str, details: dict):
        if callable(self.on_alert):
            try:
                self.on_alert(issue_type, details)
            except Exception:
                # callbacks devem ser resilientes — evitamos que quebrem a thread
                pass
