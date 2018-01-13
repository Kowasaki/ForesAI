from inference.detect import detect_camera_stream

# device_path, show_stream, write_output, NUM_CLASSES, PATH_TO_CKPT, pbtxt
detect_camera_stream(0,
                     True,
                     False,
                     90,
                     "./inference_graphs/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb",
                     "./data/mscoco_label_map.pbtxt",
                     usage_check = True)
