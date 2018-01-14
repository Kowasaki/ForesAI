from inference.detect import detect_video

# Write your video path here
your_video = "./20171208_122045.mp4"

# Input: video, number of classes for model, frozen tensorflow graph, labels
detect_video(your_video,
             90,
             "./inference_graphs/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb",
             "./data/mscoco_label_map.pbtxt")
