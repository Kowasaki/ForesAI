import time
import os
import psutil

class Timer:
    def __init__(self):
        self.init_time = time.time()

    def get_elapsed_time(self):
        curr = time.time()
        self.elapsed = curr - self.init_time
        return self.elapsed
    
    def print_elapsed_time(self):
        elapsed = self.get_elapsed_time()
        print("Time: {} seconds elapsed".format(self.elapsed))
        return elapsed

def get_cpu_usage():
    return psutil.cpu_percent()

def get_mem_usuage():
    pid = os.getpid()
    process = psutil.Process(pid)
    return process.memory_info()[0] / float(2 ** 20)

def print_cpu_usage():
    val = get_cpu_usage()
    print("CPU Usage: {} percent".format(val))
    return val

def print_mem_usage():
    val = get_mem_usuage()
    print("Memory Usage: {} MB".format(val))
    return val