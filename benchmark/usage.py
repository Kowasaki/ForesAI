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

class UsageTracker:
    def __init__(self, timer, output_path="."):
        self.timer = timer
        self.cpu_usage_dump = ""
        self.mem_usage_dump = ""
        self.time_usage_dump = ""
        self.output_path = output_path

    def get_usage(self):
        self.cpu_usage_dump += str(print_cpu_usage()) + '\n'
        self.mem_usage_dump += str(print_mem_usage()) + '\n'
        self.time_usage_dump += str(self.timer.print_elapsed_time()) + '\n'

    def _formatting(self, usage_dump):
        # TODO: add some formatting to the string dumps
        pass

    def dump_usage(self):
        self._formatting(self.cpu_usage_dump)
        self._formatting(self.mem_usage_dump)
        self._formatting(self.time_usage_dump)

        with open(os.path.join(self.output_path, "cpu_usage.txt"), "w") as c:
            c.write(self.cpu_usage_dump)
        with open(os.path.join(self.output_path, "mem_usage.txt"), "w") as m:
            m.write(self.mem_usage_dump)
        with open(os.path.join(self.output_path, "time_usage.txt"), "w") as t:
            t.write(self.time_usage_dump) 


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

def show_usage(cpu_usage_dump, mem_usage_dump, time_usage_dump, timer):
    cpu_usage_dump += str(print_cpu_usage()) + '\n'
    mem_usage_dump += str(print_mem_usage()) + '\n'
    time_usage_dump += str(timer.print_elapsed_time()) + '\n'
    return cpu_usage_dump, mem_usage_dump, time_usage_dump