
import threading
import queue

def run_in_thread(fn, daemon=True):

    t = threading.thread(target=fn, daemon=daemon)
    t.start()
    return t