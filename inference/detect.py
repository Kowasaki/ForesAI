import numpy as np
import os

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage


def load_image_into_numpy_array_color(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_image_into_numpy_array_bw(image):
    (im_width, im_height, _) = image.shape
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def show_usage(cpu_usage_dump, mem_usage_dump, time_usage_dump, timer):
    cpu_usage_dump += str(print_cpu_usage()) + '\n'
    mem_usage_dump += str(print_mem_usage()) + '\n'
    time_usage_dump += str(timer.print_elapsed_time()) + '\n'
    return cpu_usage_dump, mem_usage_dump, time_usage_dump

def detect(config):

    if config["library"] == "tensorflow":
        from inference.tf_op import graph_prep, run_detection
        detection_graph, label_map, categories, category_index = graph_prep(config["model"]["classes"],
            config["model"]["model_path"],config["model"]["pbtxt"])

        run_detection(config["device_path"], detection_graph, label_map, categories, category_index, 
            config["show_stream"], config["show_stream"], config["write_output"], config["benchmark"])        
    