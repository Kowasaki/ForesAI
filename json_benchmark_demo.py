import json

from benchmark import visualize_benchmark
from inference.detect import detect

# Use a json as config file for setting up detection
with open("./demo_configs/benchmark_webcam.json", 'r') as j:
    config = json.load(j)
    detect(config)

# Plot the cpu and memory usage over time. If you have a nvidia gpu put "gpu = True" in the argument
visualize_benchmark.plot_resource_over_time(".")