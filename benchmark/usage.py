import os
import psutil

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