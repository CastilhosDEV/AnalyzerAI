# analyzer/monitor.py
"""
SystemMonitor - Avançado (B3-capable)
- Monitora CPU, RAM, Disco e processos
- Envia eventos para GUI via queue: ('monitor_alert', detail), ('monitor_suggest', detail), ('monitor_action', detail), ('monitor_log', msg)
- Autonomy control (B1/B2/B3) via analyzer_settings.json
"""

import os
import time
import psutil
import json
import logging
import threading
import shutil
import subprocess
from typing import Optional, Any, Dict, Tuple, List

logger = logging.getLogger("SystemMonitor")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "analyzer_settings.json")
if not os.path.exists(SETTINGS_PATH):
    raise FileNotFoundError("analyzer_settings.json missing for monitor.")

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

DIAG = SETTINGS.get("diagnostic_mode", {})
AUTONOMY = DIAG.get("autonomy", "B2")
ALLOW_REPAIRS = DIAG.get("allow_system_repairs", False)

class SystemMonitor:
    def __init__(self, queue=None, interval: float = None):
        self.queue = queue
        self.interval = interval if interval is not None else DIAG.get("auto_scan_interval", 8)
        self.cpu_threshold = DIAG.get("cpu_threshold", 90)
        self.mem_threshold = DIAG.get("mem_threshold", 85)
        self.disk_threshold = DIAG.get("disk_threshold", 92)
        self._running = False
        self.action_history: List[Dict[str,Any]] = []

    def start_monitoring(self):
        logger.info(f"[Monitor] starting monitor (autonomy={AUTONOMY}) interval={self.interval}s")
        self._running = True
        time.sleep(0.5)
        while self._running:
            try:
                self._check_once()
            except Exception as e:
                logger.exception(f"[Monitor] loop error: {e}")
                self._emit("monitor_log", f"monitor loop error: {e}")
            time.sleep(self.interval)

    def stop(self):
        self._running = False

    def _check_once(self):
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage(os.path.expanduser("~")).percent
        top = self._get_top_processes(6)

        self._emit("monitor_log", f"heartbeat CPU {cpu}% MEM {mem}% DISK {disk}%")

        if cpu >= self.cpu_threshold or mem >= self.mem_threshold or disk >= self.disk_threshold:
            detail = {"cpu": cpu, "mem": mem, "disk": disk, "top_procs": top}
            self._emit("monitor_alert", detail)
            # act based on autonomy
            if AUTONOMY == "B3":
                # automatic mitigation
                self._mitigate_critical(detail)
        else:
            # suggestions
            if disk >= (self.disk_threshold - 6):
                self._emit("monitor_suggest", {"suggest": "clean_temp", "disk": disk, "top_procs": top})
            if mem >= (self.mem_threshold - 6) and AUTONOMY == "B3":
                self._auto_free_memory(top)

    def _get_top_processes(self, n=5):
        procs = []
        for p in psutil.process_iter(attrs=["pid","name","cpu_percent","memory_percent"]):
            try:
                info = p.info
                # ensure cpu_percent populated
                if info.get("cpu_percent") is None:
                    info["cpu_percent"] = p.cpu_percent(interval=None)
                if info.get("memory_percent") is None:
                    info["memory_percent"] = p.memory_percent()
                procs.append(info)
            except Exception:
                continue
        procs_sorted = sorted(procs, key=lambda x: (x.get("cpu_percent",0) or 0) + (x.get("memory_percent",0) or 0), reverse=True)
        return procs_sorted[:n]

    def _mitigate_critical(self, detail: Dict[str,Any]):
        top = detail.get("top_procs", [])
        if top:
            pid = top[0].get("pid")
            ok, msg = self.kill_process(pid)
            self._record_action("kill_top", {"pid": pid, "ok": ok, "msg": msg})
            self._emit("monitor_action", {"action": "kill_top", "pid": pid, "ok": ok, "msg": msg})
        freed = self.free_memory()
        self._record_action("free_memory", {"result": freed})
        self._emit("monitor_action", {"action":"free_memory","result":freed})
        if detail.get("disk") and detail.get("disk") >= (self.disk_threshold - 2):
            res = self.clean_temp_folder()
            self._record_action("clean_temp", {"removed": len(res.get("removed",[])),"failed": len(res.get("failed",[]))})
            self._emit("monitor_action", {"action":"clean_temp","result":res})
        if ALLOW_REPAIRS:
            repairs = self._maybe_run_repairs()
            self._record_action("repairs", {"msgs": repairs})
            self._emit("monitor_action", {"action":"repairs","msgs":repairs})

    def _auto_free_memory(self, top_procs):
        for p in top_procs:
            name = (p.get("name") or "").lower()
            pid = p.get("pid")
            if pid == os.getpid():
                continue
            if any(x in name for x in ("system","svchost","explorer","wininit","python")):
                continue
            ok, msg = self.kill_process(pid)
            self._record_action("kill_auto", {"pid": pid, "ok": ok, "msg": msg})
            self._emit("monitor_action", {"action":"kill_auto","pid":pid,"ok":ok,"msg":msg})
            break

    def kill_process(self, pid: int) -> Tuple[bool, str]:
        """
        Termina processo de forma segura. Retorna (ok, mensagem).
        Esta assinatura usa typing.Tuple (correção do erro de hint).
        """
        try:
            p = psutil.Process(pid)
            name = p.name()
            p.terminate()
            gone, alive = psutil.wait_procs([p], timeout=3)
            if alive:
                try:
                    p.kill()
                except Exception:
                    pass
            return True, f"Processo {name} (PID {pid}) terminado."
        except psutil.NoSuchProcess:
            return False, f"Processo {pid} não encontrado."
        except Exception as e:
            return False, f"Falha ao terminar PID {pid}: {e}"

    def clean_temp_folder(self, paths: Optional[List[str]] = None) -> Dict[str,Any]:
        if paths is None:
            if os.name == "nt":
                paths = [os.environ.get("TEMP",""), os.path.join(os.path.expanduser("~"), "AppData","Local","Temp")]
            else:
                paths = ["/tmp"]
        removed = []
        failed = []
        for base in paths:
            if not base or not os.path.exists(base):
                continue
            try:
                for name in os.listdir(base):
                    fp = os.path.join(base, name)
                    try:
                        if os.path.isfile(fp) or os.path.islink(fp):
                            os.remove(fp)
                            removed.append(fp)
                        elif os.path.isdir(fp):
                            shutil.rmtree(fp, ignore_errors=True)
                            removed.append(fp)
                    except Exception as e:
                        failed.append((fp, str(e)))
            except Exception as e:
                failed.append((base, str(e)))
        return {"removed": removed, "failed": failed}

    def free_memory(self) -> Dict[str,Any]:
        # heuristic: on linux attempt drop caches (requires root) otherwise no-op; on windows call GC hint
        result = {"action": None, "ok": False, "detail": None}
        try:
            if os.name == "posix":
                try:
                    subprocess.run(["sync"], check=False)
                    # try writing to drop_caches; ignore failure
                    with open("/proc/sys/vm/drop_caches", "w") as f:
                        f.write("3\n")
                    result["action"] = "drop_caches"
                    result["ok"] = True
                except Exception as e:
                    result["detail"] = f"drop_caches failed: {e}"
            elif os.name == "nt":
                # Windows: attempt small memory heuristics - not destructive
                result["action"] = "windows_noop"
                result["ok"] = True
            else:
                result["action"] = "noop"
                result["ok"] = False
        except Exception as e:
            result["detail"] = str(e)
            result["ok"] = False
        return result

    def _maybe_run_repairs(self) -> List[str]:
        msgs = []
        if not ALLOW_REPAIRS:
            msgs.append("system repairs disabled.")
            return msgs
        try:
            if os.name == "nt":
                try:
                    subprocess.run(["sfc", "/scannow"], check=False)
                    msgs.append("sfc /scannow invoked (output in system logs).")
                except Exception as e:
                    msgs.append(f"sfc error: {e}")
            else:
                msgs.append("no automatic destructive repairs on POSIX.")
        except Exception as e:
            msgs.append(str(e))
        return msgs

    def _emit(self, typ: str, payload: Any):
        try:
            if self.queue:
                self.queue.put((typ, payload))
        except Exception:
            logger.exception("failed to emit event")

    def _record_action(self, name: str, detail: dict):
        self.action_history.append({"ts": time.time(), "action": name, "detail": detail})
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
