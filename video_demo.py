from inference.detect import detect_video

# Input: video, number of classes for model, frozen tensorflow graph, labels
detect_video("./20171208_122045.mp4",
             90,
             "./inference_graphs/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb",
             "./data/mscoco_label_map.pbtxt")
