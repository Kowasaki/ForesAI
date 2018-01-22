import cv2
import logging
import numpy as np
import os
import tensorflow as tf

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage
from inference.box_op import Box, parse_tf_output
from inference.detect import detect, show_usage
from tf_object_detection.utils import label_map_util
from tf_object_detection.utils import visualization_utils as vis_util

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

    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise Exception("Video not found!")
        
    c = int(vid.get(3))  
    r = int(vid.get(4)) 
    logger.debug("Frame width: {} height: {}".format(r,c))
    
    if write_output:
        trackedVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (c,r))
        record = open("record.txt", "w")

    count = 0

    # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Read video frame by frame and perform inference
            while(vid.isOpened()):
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                logger.debug("frame {}".format(count))
                retval, curr_frame = vid.read()

                if not retval:
                    logger.info("Video ending at frame {}".format(count))
                    break

                if show_window:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                curr_frame_expanded = np.expand_dims(curr_frame, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: curr_frame_expanded})

                if usage_check:
                    logger.info("Frame {}".format(count))
                    cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
                        mem_usage_dump, time_usage_dump, timer)
                
                topleft_pts = []
                widths = []
                heights = []
                labels = []

                # get boxes that pass the min requirements and their pixel coordinates
                (r,c,_) = curr_frame.shape
                logger.debug("image row:{}, col:{}".format(r,c))
                
                filtered_boxes = parse_tf_output(curr_frame.shape, boxes, scores, classes)

                logger.debug("".join([str(i) for i in filtered_boxes]))

                # TODO: Send the detected info to other systems every frame
                
                if write_output:
                    record.write(str(count)+"\n")            
                    for i in range(len(filtered_boxes)):
                        record.write("{}\n".format(str(filtered_boxes[i])))

                # Visualization of the results of a detection.
                curr_frame = curr_frame[...,::-1] #flip bgr back to rgb (Thanks OpenCV)
                if visualize:
                    drawn_img = vis_util.visualize_boxes_and_labels_on_image_array(
                        curr_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    if show_window:
                        window_name = "stream"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.imshow(window_name,drawn_img[...,::-1])
                
                    if write_output:
                        trackedVideo.write(drawn_img[...,::-1])
                
                count += 1

    if usage_check:
        logger.info("Total Time elapsed: {} seconds".format(timer.get_elapsed_time()))
        with open("cpu_usage.txt", "w") as c:
            c.write(cpu_usage_dump)
        with open("mem_usage.txt", "w") as m:
            m.write(mem_usage_dump)
        with open("time_usage.txt", "w") as t:
            t.write(time_usage_dump) 

    vid.release()

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

