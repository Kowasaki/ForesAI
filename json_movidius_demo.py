import json

from inference.detect import detect

# Use a json as config file for setting up detection
with open("./demo_configs/jetson_movidius.json", 'r') as j:
    config = json.load(j)
    detect(config)

