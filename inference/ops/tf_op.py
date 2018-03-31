import copy
import cv2
import logging
import numpy as np
import os
import signal
import tensorflow as tf
import time

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage, show_usage
from inference.detect import detect
from tensorflow.core.framework import graph_pb2
from tf_object_detection.utils import label_map_util
from tf_object_detection.utils import ops as utils_ops 
from utils.box_op import Box, parse_tf_output
from utils.fps import FPS
from utils.videostream import WebcamVideoStream
from utils.visualize import overlay


def _node_name(n):

  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

def load_model(PATH_TO_CKPT):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    return detection_graph

# source: https://github.com/GustavZ/realtime_object_detection.git under object_detection.py
def load_split_model(PATH_TO_CKPT):
    # load a frozen Model and split it into GPU and CPU graphs
    input_graph = tf.Graph()
    with tf.Session(graph=input_graph):
        score = tf.placeholder(tf.float32, shape=(None, 1917, 90), name="Postprocessor/convert_scores")
        expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
        for node in input_graph.as_graph_def().node:
            if node.name == "Postprocessor/convert_scores":
                score_def = node
            if node.name == "Postprocessor/ExpandDims_1":
                expand_def = node
                
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
        
            edges = {}
            name_to_node_map = {}
            node_seq = {}
            seq = 0
            for node in od_graph_def.node:
                n = _node_name(node.name)
                name_to_node_map[n] = node
                edges[n] = [_node_name(x) for x in node.input]
                node_seq[n] = seq
                seq += 1
        
            for d in dest_nodes:
                assert d in name_to_node_map, "%s is not in graph" % d
        
            nodes_to_keep = set()
            next_to_visit = dest_nodes[:]
            while next_to_visit:
                n = next_to_visit[0]
                del next_to_visit[0]
                if n in nodes_to_keep:
                    continue
                nodes_to_keep.add(n)
                next_to_visit += edges[n]
        
            nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
        
            nodes_to_remove = set()
            for n in node_seq:
                if n in nodes_to_keep_list: continue
                nodes_to_remove.add(n)
            nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])
        
            keep = graph_pb2.GraphDef()
            for n in nodes_to_keep_list:
                keep.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            remove = graph_pb2.GraphDef()
            remove.node.extend([score_def])
            remove.node.extend([expand_def])
            for n in nodes_to_remove_list:
                remove.node.extend([copy.deepcopy(name_to_node_map[n])])
    
            with tf.device('/gpu:0'):
                tf.import_graph_def(keep, name='')
            with tf.device('/cpu:0'):
                tf.import_graph_def(remove, name='')

    return detection_graph, score, expand

def load_label_map(NUM_CLASSES, pbtxt):

    # Loading Label Map
    label_map = label_map_util.load_labelmap(pbtxt)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
        max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return label_map, categories, category_index

def run_detection(video_path,
                  detection_graph, 
                  label_map, 
                  categories, 
                  category_index, 
                  show_window,
                  visualize, 
                  write_output,
                  ros_enabled, 
                  usage_check,
                  graph_trace_enabled = False,
                  score_node = None,
                  expand_node = None):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = tf.ConfigProto(allow_soft_placement=True)

    labels_per_frame = []
    boxes_per_frame = []
    cpu_usage_dump = ""
    mem_usage_dump = ""
    time_usage_dump = ""

    if ros_enabled:
        from utils.ros_op import DetectionPublisher, CameraSubscriber
        pub = DetectionPublisher()
        sub = CameraSubscriber()

    if graph_trace_enabled:
        from tensorflow.python.client import timeline

    if usage_check:
        timer = Timer()
        logger.info("Initial startup")
        cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
            mem_usage_dump, time_usage_dump, timer)
    
    if ros_enabled:
        if not sub.is_running():
            return Exception("[ERROR: Camera Node not running]")
    else:
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
            options = None
            run_metadata = None
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

            # Using the split model hack
            if score_node is not None and expand_node is not None:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')                
            
            if usage_check:
                fps = FPS().start()
            
            if graph_trace_enabled:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            # Read video frame by frame and perform inference
            while(vid.is_running()):
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
                    if score_node is None and expand_node is None:
                        (boxes, scores, classes) = sess.run(
                            [detection_boxes, detection_scores, detection_classes],
                            feed_dict={image_tensor: curr_frame_expanded}, 
                            options=options,
                            run_metadata=run_metadata)
                    else:
                        # Split Detection in two sessions.
                        (score, expand) = sess.run(
                            [score_out, expand_out], 
                            feed_dict={image_tensor: curr_frame_expanded})
                        (boxes, scores, classes) = sess.run(
                            [detection_boxes, detection_scores, detection_classes],
                            feed_dict={score_in:score, expand_in: expand}) 
                    end = time.time()

                    if usage_check:
                        fps.update()
                        logger.info("Session run time: {:.4f}".format(end - start))
                        logger.info("Frame {}".format(count))
                        cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
                            mem_usage_dump, time_usage_dump, timer)
                    
                    if graph_trace_enabled:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open('graph_timeline.json' , 'w') as f:
                            f.write(chrome_trace)
                    
                    (r,c,_) = curr_frame.shape
                    logger.debug("image height:{}, width:{}".format(r,c))
                    # get boxes that pass the min requirements and their pixel coordinates
                    filtered_boxes = parse_tf_output(curr_frame.shape, boxes, scores, classes)


                    if ros_enabled:
                    # TODO: Send the detected info to other systems every frame
                        logger.info("Publishing bboxes")
                        logger.info("".join([str(i) for i in filtered_boxes]))
                        pub.send_boxes(filtered_boxes)
                        

                    
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
                    else:
                        logger.info("".join([str(i) for i in filtered_boxes]))

                    
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

def run_mask_detection(video_path,
                       detection_graph, 
                       label_map, 
                       categories, 
                       category_index, 
                       show_window,
                       visualize, 
                       write_output,
                       ros_enabled, 
                       usage_check,
                       graph_trace_enabled = False,
                       score_node = None,
                       expand_node = None):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from tf_object_detection.utils import ops as utils_ops 
    from PIL import Image
    from tf_object_detection.utils import visualization_utils as vis_util

    config = tf.ConfigProto(allow_soft_placement=True)

    labels_per_frame = []
    boxes_per_frame = []
    cpu_usage_dump = ""
    mem_usage_dump = ""
    time_usage_dump = ""

    if ros_enabled:
        from utils.ros_op import DetectionPublisher, CameraSubscriber
        pub = DetectionPublisher()
        sub = CameraSubscriber()
    
    if graph_trace_enabled:
        from tensorflow.python.client import timeline

    if usage_check:
        timer = Timer()
        logger.info("Initial startup")
        cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
            mem_usage_dump, time_usage_dump, timer)
    
    if ros_enabled:
        if not sub.is_running():
            return Exception("[ERROR: Camera Node not running]")
    else:
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
            options = None
            run_metadata = None
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, r, c)
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Using the split model hack
            if score_node is not None and expand_node is not None:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')                
            
            if usage_check:
                fps = FPS().start()

            if graph_trace_enabled:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            # Read video frame by frame and perform inference
            while(vid.is_running()):
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

                    # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    curr_frame_expanded = np.expand_dims(curr_frame, axis=0)

                    # Actual detection.
                    start = time.time()
                    if score_node is None and expand_node is None:
                        output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: curr_frame_expanded},
                            options=options,
                            run_metadata=run_metadata)
                    else:
                        raise Exception("Split model not supported for mask")
                        # Split Detection in two sessions.
                        # (score, expand) = sess.run(
                        #     [score_out, expand_out], 
                        #     feed_dict={image_tensor: curr_frame_expanded})
                        # (boxes, scores, classes) = sess.run(
                        #     [detection_boxes, detection_scores, detection_classes],
                        #     feed_dict={score_in:score, expand_in: expand}) 
                    end = time.time()

                    boxes = output_dict['detection_boxes']
                    scores = output_dict['detection_scores']
                    classes =output_dict['detection_classes']

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    output_dict['detection_masks'] = output_dict['detection_masks'][0] 

                    logger.info(output_dict['detection_masks'].shape)                   

                    if usage_check:
                        fps.update()
                        logger.info("Session run time: {:.4f}".format(end - start))
                        logger.info("Frame {}".format(count))
                        cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
                            mem_usage_dump, time_usage_dump, timer)
                    
                    if graph_trace_enabled:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open('graph_timeline.json' , 'w') as f:
                            f.write(chrome_trace)

                    (r,c,_) = curr_frame.shape
                    logger.debug("image height:{}, width:{}".format(r,c))
                    # get boxes that pass the min requirements and their pixel coordinates
                    filtered_boxes = parse_tf_output(curr_frame.shape, boxes, scores, classes)


                    if ros_enabled:
                    # TODO: Send the detected info to other systems every frame
                        logger.info("Publishing bboxes")
                        logger.info("".join([str(i) for i in filtered_boxes]))
                        pub.send_boxes(filtered_boxes)
                        

                    
                    if write_output:
                        record.write(str(count)+"\n")            
                        for i in range(len(filtered_boxes)):
                            record.write("{}\n".format(str(filtered_boxes[i])))

                    # Visualization of the results of a detection.
                    if visualize:
                        # drawn_img = overlay(curr_frame, category_index, filtered_boxes)
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            curr_frame,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        if show_window:
                            window_name = "stream"
                            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            # cv2.imshow(window_name,drawn_img)
                            cv2.imshow(window_name,curr_frame)

                        if write_output:
                            # trackedVideo.write(drawn_img)
                            trackedVideo.write(curr_frame)
                    else:
                        logger.info("".join([str(i) for i in filtered_boxes]))

                    
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
    
    detection_graph = load_model(PATH_TO_CKPT)
    label_map, categories, category_index = load_label_map(NUM_CLASSES, pbtxt)
    run_detection(video_path, detection_graph, label_map, categories, category_index, False, True, True, False, False)

def detect_camera_stream(device_path,
                         show_stream,
                         write_output,
                         NUM_CLASSES,
                         PATH_TO_CKPT,
                         pbtxt,
                         ros_enabled,
                         usage_check = False):

    detection_graph = load_model(PATH_TO_CKPT)
    label_map, categories, category_index = load_label_map(NUM_CLASSES, pbtxt)
    run_detection(device_path, detection_graph, label_map, categories, category_index, show_stream, 
        show_stream, write_output, ros_enabled, usage_check)

