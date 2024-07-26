import multiprocessing
import os
import time
import atexit
import sys

exception_occurred = False

def global_exception_handler(exctype, value, traceback):
    global exception_occurred
    exception_occurred = True
    print("Global exception handler called.")
    sys.__excepthook__(exctype, value, traceback)

class MonitorProcess(multiprocessing.Process):

    def __init__(self, main_pid):
        super().__init__()
        self.main_pid = main_pid
        self.exit_event = multiprocessing.Event()
        self.crashed = multiprocessing.Event()

    def run(self):
        os.setsid()

        print('GEEZ START CHECKING STATUS ON PID: ', os.getpid())

        while not self.exit_event.wait(1):
            try:
                # Signal 0 does not kill the process but performs error checking
                os.kill(self.main_pid, 0)
                print('GEEZ MAIN PROCESS IS ALIVE!')
            except OSError:
                print('GEEZ MAIN PROCESS IS NOT ALIVE!')
                break


        print('Monitor process exiting gracefully.')

    def stop(self):
        self.exit_event.set()

    def crash(self):
        self.crashed.set()
        self.exit_event.set()

class MainProcess:
    def run(self):
        atexit.register(self.atexit_handler)

        self.monitor_process = MonitorProcess(main_pid=os.getpid())
        self.monitor_process.start()

        print(f"Parent process: PID={os.getpid()}, Child PID={self.monitor_process.pid}")
        time.sleep(3)
        print("Parent process is exiting...")

        raise ValueError()


    def atexit_handler(self):
        print("GEEZ I AM CALLING ATEXIT HANDLER!")

        global exception_occured
        if exception_occurred:
            print("GEEZ MOLY EXCEPTION OCCURED!")
        else:
            print("GEEZ COME ON THIS IS JUST A NORMAL FINISH!")

        # self.monitor_process.crash()


if __name__ == "__main__":
    sys.excepthook = global_exception_handler

    process = MainProcess()
    process.run()
