import json

from benchmark import visualize_benchmark
from inference.detect import detect

# Use a json as config file for setting up detection
with open("./demo_configs/webcam_erfnet.json", 'r') as j:
    config = json.load(j)
    detect(config)

visualize_benchmark.plot_resource_over_time(".")