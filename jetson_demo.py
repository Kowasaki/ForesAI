from inference.detect import detect_camera_stream
from benchmark import visualize_benchmark

# Put your TensorFlow model checkpoint and label mapping here
ckpt = "./inference_graphs/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
pbtxt = "./data/mscoco_label_map.pbtxt"

# device_path, show_stream, write_output, NUM_CLASSES, PATH_TO_CKPT, pbtxt
# press the "q" key on the display window to exit
# Uses the camera from the Jetson TX2 development kit
detect_camera_stream("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",
                     True,
                     False,
                     90,
                     ckpt,
                     pbtxt,
                     usage_check = True)

# Plot the cpu and memory usage over time. If you have a nvidia gpu put "gpu = True" in the argument
visualize_benchmark.plot_resource_over_time("./jetson_stats")
