import time
import signal

class OnTime():
    def __init__(self, interval):
        self.interval = interval 
        self.last_run = None
    def __call__(self):
        now = time.time()
        if self.last_run is None:
            self.last_run = now
        if now - self.last_run > self.interval:
            self.last_run = now
            return True
        return False

class Breakpoint():
    def __init__(self):
        import signal
        self.break_point = False

        def handler(signum, frame):
            self.break_point = True
        signal.signal(signal.SIGQUIT, handler)

    def __call__(self):
        if self.break_point:
            import pdb
            self.break_point = False
            pdb.set_trace()
