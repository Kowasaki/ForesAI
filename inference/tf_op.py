import cv2
import logging
import numpy as np
import os
import signal
import tensorflow as tf
import time

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage, show_usage
from inference.detect import detect
from tf_object_detection.utils import label_map_util
from utils.box_op import Box, parse_tf_output
from utils.fps import FPS
from utils.videostream import WebcamVideoStream
from utils.visualize import overlay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def graph_prep(NUM_CLASSES,
               PATH_TO_CKPT,
               pbtxt):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading Label Map
    label_map = label_map_util.load_labelmap(pbtxt)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
        max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, label_map, categories, category_index

def run_detection(video_path,
                  detection_graph, 
                  label_map, 
                  categories, 
                  category_index, 
                  show_window,
                  visualize, 
                  write_output, 
                  usage_check):

    config = tf.ConfigProto()

    labels_per_frame = []
    boxes_per_frame = []
    cpu_usage_dump = ""
    mem_usage_dump = ""
    time_usage_dump = ""

    if usage_check:
        timer = Timer()
        logger.info("Initial startup")
        cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
            mem_usage_dump, time_usage_dump, timer)

    vid = WebcamVideoStream(src = video_path).start()

    r, c = vid.get_dimensions()

    logger.debug("Frame width: {} height: {}".format(r,c))
    
    if write_output:
        trackedVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (c,r))
        record = open("record.txt", "w")

    count = 0

    # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config = config) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            
            if usage_check:
                fps = FPS().start()

            # Read video frame by frame and perform inference
            while(vid.stream.isOpened()):
                try:
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    logger.debug("Frame {}".format(count))
                    retval, curr_frame = vid.read()

                    if not retval:
                        logger.info("Video ending at frame {}".format(count))
                        break

                    if show_window:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    curr_frame_expanded = np.expand_dims(curr_frame, axis=0)
                    curr_frame_expanded = np.int8(curr_frame_expanded)

                    # Actual detection.
                    start = time.time()
                    (boxes, scores, classes) = sess.run(
                        [detection_boxes, detection_scores, detection_classes],
                        feed_dict={image_tensor: curr_frame_expanded})
                    end = time.time()
                    logger.info("Session run time: {:.4f}".format(end - start))

                    if usage_check:
                        fps.update()
                        logger.info("Frame {}".format(count))
                        cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
                            mem_usage_dump, time_usage_dump, timer)
                    
                    (r,c,_) = curr_frame.shape
                    logger.debug("image height:{}, width:{}".format(r,c))
                    # get boxes that pass the min requirements and their pixel coordinates
                    filtered_boxes = parse_tf_output(curr_frame.shape, boxes, scores, classes)

                    logger.debug("".join([str(i) for i in filtered_boxes]))

                    # TODO: Send the detected info to other systems every frame
                    
                    if write_output:
                        record.write(str(count)+"\n")            
                        for i in range(len(filtered_boxes)):
                            record.write("{}\n".format(str(filtered_boxes[i])))

                    # Visualization of the results of a detection.
                    if visualize:
                        drawn_img = overlay(curr_frame, category_index, filtered_boxes)
                        if show_window:
                            window_name = "stream"
                            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            cv2.imshow(window_name,drawn_img)
                    
                        if write_output:
                            trackedVideo.write(drawn_img)
                    
                    count += 1

                except KeyboardInterrupt:
                    logger.info("Ctrl + C Pressed. Attempting graceful exit")
                    break

    if usage_check:
        fps.stop()
        logger.info("[USAGE] elasped time: {:.2f}".format(fps.elapsed()))
        logger.info("[USAGE] approx. FPS: {:.2f}".format(fps.fps()))
        logger.info("[USAGE] inferenced frames: {}".format(fps.get_frames()))
        logger.info("[USAGE] raw frames: {}".format(vid.get_raw_frames()))
        logger.info("[USAGE] Total Time elapsed: {:.2f} seconds".format(timer.get_elapsed_time()))
        with open("cpu_usage.txt", "w") as c:
            c.write(cpu_usage_dump)
        with open("mem_usage.txt", "w") as m:
            m.write(mem_usage_dump)
        with open("time_usage.txt", "w") as t:
            t.write(time_usage_dump) 

    vid.stop()

    logger.debug("Result: {} frames".format(count))
    
    if visualize:
        cv2.destroyAllWindows()

    if write_output:
        record.close()
        trackedVideo.release()

    return labels_per_frame, boxes_per_frame

def detect_video(video_path,
                 NUM_CLASSES,
                 PATH_TO_CKPT,
                 pbtxt):

    detection_graph, label_map, categories, category_index = graph_prep(NUM_CLASSES,PATH_TO_CKPT,pbtxt)
    run_detection(video_path, detection_graph, label_map, categories, category_index, False, True, True, False)

def detect_camera_stream(device_path,
                         show_stream,
                         write_output,
                         NUM_CLASSES,
                         PATH_TO_CKPT,
                         pbtxt,
                         usage_check = False):

    detection_graph, label_map, categories, category_index = graph_prep(NUM_CLASSES,PATH_TO_CKPT,pbtxt)
    run_detection(device_path, detection_graph, label_map, categories, category_index, show_stream, 
        show_stream, write_output, usage_check)

