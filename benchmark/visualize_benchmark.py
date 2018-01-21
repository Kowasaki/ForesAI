import matplotlib.pyplot as plt
import os.path

def parse_benchmark_values(src):
    val_list = []

    with open(src, 'r') as source:
        for line in source:
            val_list.append(float(line))
    
    return val_list

def plot_resource_over_time(path, cpu = True, mem = True, gpu = False):

    plots = 0 
    time_src = os.path.join(path,"time_usage.txt")
    time_list = parse_benchmark_values(time_src)

    print("Startup Time: {} seconds".format(float(time_list[1])))
    print("{} Frame per Second".format(float(len(time_list)-1)/float(time_list[-1])))

    if cpu:
        cpu_src = os.path.join(path,"cpu_usage.txt")
        cpu_list = parse_benchmark_values(cpu_src)
        plots += 1
        plt.figure(plots)
        plt.plot(time_list, cpu_list, 'b')
        plt.xlabel("Seconds")
        plt.ylabel("% Usage")
        plt.title("CPU usage over time")

    if mem:
        mem_src = os.path.join(path,"mem_usage.txt")
        mem_list = parse_benchmark_values(mem_src)
        plots += 1
        plt.figure(plots)
        plt.plot(time_list, mem_list, 'r')
        plt.xlabel("Seconds")
        plt.ylabel("MB")
        plt.title("Memory usage over time")

    if gpu:
        gpu_src = os.path.join(path,"gpu_usage.txt")
        gpu_list = parse_benchmark_values(gpu_src)
        plots += 1
        plt.figure(plots)
        plt.plot(time_list, gpu_list, 'g')
        plt.xlabel("Seconds")
        plt.ylabel("% Usage")
        plt.title("GPU usage over time")

    plt.show()


