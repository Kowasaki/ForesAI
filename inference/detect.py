import cv2
import numpy as np
import os
import tensorflow as tf

from benchmark.usage import get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array_color(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_image_into_numpy_array_bw(image):
    (im_width, im_height, _) = image.shape
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def show_usage(cpu_usage_dump, mem_usage_dump):
    cpu_usage_dump += str(print_cpu_usage()) + '\n'
    mem_usage_dump += str(print_mem_usage()) + '\n'
    return cpu_usage_dump, mem_usage_dump

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

def detect_video(video_path,
                 NUM_CLASSES,
                 PATH_TO_CKPT,
                 pbtxt):

    # # Load a (frozen) Tensorflow model into memory.
    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')

    # # Loading Label Map
    # label_map = label_map_util.load_labelmap(pbtxt)
    # categories = label_map_util.convert_label_map_to_categories(label_map, 
    #     max_num_classes=NUM_CLASSES, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    detection_graph, label_map, categories, category_index = graph_prep(NUM_CLASSES,PATH_TO_CKPT,pbtxt)

    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise Exception("Video not found!")
        
    c = int(vid.get(3))  
    r = int(vid.get(4)) 
    # print(c,r)
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
                # print("frame {}".format(count))
                record.write(str(count)+"\n")
                retval, curr_frame = vid.read()

                if not retval:
                    print("Video ending at frame {}".format(count))
                    break


                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                curr_frame_expanded = np.expand_dims(curr_frame, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: curr_frame_expanded})


                # Need manual input (in meters)
                topleft_pts = []
                widths = []
                heights = []
                labels = []

                # get boxes that pass the min requirements and their pixel coordinates
                (r,c,_) = curr_frame.shape
                # print("image row:{}, col:{}".format(r,c))
                for i in range(len(boxes[0])):
                    if scores[0][i] > 0.5:
                        topleft = (int(r*boxes[0][i][0]),int(c*boxes[0][i][1]))
                        topright = (int(r*boxes[0][i][0]),int(c*boxes[0][i][3]))
                        botleft = (int(r*boxes[0][i][2]),int(c*boxes[0][i][1]))
                        botright = (int(r*boxes[0][i][2]),int(c*boxes[0][i][3]))
                        height = int(r*boxes[0][i][2] - r*boxes[0][i][0])
                        width = int(c*boxes[0][i][3] - c*boxes[0][i][1])

                        labels.append(classes[0][i])
                        topleft_pts.append(topleft)
                        widths.append(width)
                        heights.append(height)

                # # assign real-world heights using obj class
                # meter_heights = assign_heights(labels)

                # x_mins, x_maxs, y_mins, y_maxs, zs = convert_pixel_to_metric(r, c, topleft_pts, 
                #                                         meter_heights[:len(heights)], widths, heights)

                # for i in range(len(zs)):
                #     record.write(str(x_mins[i])+ "," + str(x_maxs[i])+ "," + str(y_mins[i])+ "," + 
                #         str(y_maxs[i])+ "," + str(zs[i])+ "\n")       
        
                for i in range(len(topleft_pts)):
                    record.write(str(topleft_pts[i][0])+ "," + str(topleft_pts[i][1])+ "," + str(heights[i])+ "," + 
                        str(widths[i]) + "\n")  
                # Visualization of the results of a detection.
                curr_frame = curr_frame[...,::-1] #flip bgr back to rgb (Thanks OpenCV)
                drawn_img = vis_util.visualize_boxes_and_labels_on_image_array(
                    curr_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                trackedVideo.write(drawn_img[...,::-1])
                count += 1
    vid.release()
    record.close()

    print("Result: {} frames".format(count))

    trackedVideo.release()



def detect_camera_stream(device_path,
                         show_stream,
                         write_output,
                         NUM_CLASSES,
                         PATH_TO_CKPT,
                         pbtxt,
                         usage_check = False):
    
    cpu_usage_dump = ""
    mem_usage_dump = ""

    if usage_check:
        print("Initial startup")
        cpu_usage_dump, mem_usage_dump = show_usage(cpu_usage_dump, mem_usage_dump)

    detection_graph, label_map, categories, category_index = graph_prep(NUM_CLASSES,PATH_TO_CKPT,pbtxt)
    vid = cv2.VideoCapture(device_path)

    if not vid.isOpened():
        raise Exception("Camera device not found!")
        
    c = int(vid.get(3))  
    r = int(vid.get(4)) 
    # print(c,r)
    if write_output:
        trackedVideo = cv2.VideoWriter('stream.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (c,r))
        record = open("stream_record.txt", "w")

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
                # print("frame {}".format(count))
                retval, curr_frame = vid.read()

                if not retval:
                    print("Video ending at frame {}".format(count))
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                curr_frame_expanded = np.expand_dims(curr_frame, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: curr_frame_expanded})

                if usage_check:
                    print("Frame {}".format(count))
                    cpu_usage_dump, mem_usage_dump = show_usage(cpu_usage_dump, mem_usage_dump)

                if write_output:
                    record.write(str(count)+"\n")
                    topleft_pts = []
                    widths = []
                    heights = []
                    labels = []

                    # get boxes that pass the min requirements and their pixel coordinates
                    (r,c,_) = curr_frame.shape
                    for i in range(len(boxes[0])):
                        if scores[0][i] > 0.5:
                            topleft = (int(r*boxes[0][i][0]),int(c*boxes[0][i][1]))
                            topright = (int(r*boxes[0][i][0]),int(c*boxes[0][i][3]))
                            botleft = (int(r*boxes[0][i][2]),int(c*boxes[0][i][1]))
                            botright = (int(r*boxes[0][i][2]),int(c*boxes[0][i][3]))
                            height = int(r*boxes[0][i][2] - r*boxes[0][i][0])
                            width = int(c*boxes[0][i][3] - c*boxes[0][i][1])
                            labels.append(classes[0][i])
                            topleft_pts.append(topleft)
                            widths.append(width)
                            heights.append(height)

                    for i in range(len(topleft_pts)):
                        record.write(str(topleft_pts[i][0])+ "," + str(topleft_pts[i][1])+ "," + str(heights[i])+ "," + 
                            str(widths[i]) + "\n")  

                # Visualization of the results of a detection.
                curr_frame = curr_frame[...,::-1] #flip bgr back to rgb (Thanks OpenCV)
                if show_stream:
                    drawn_img = vis_util.visualize_boxes_and_labels_on_image_array(
                        curr_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    window_name = "stream"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name,drawn_img[...,::-1])


                if write_output:
                    trackedVideo.write(drawn_img[...,::-1])

                count += 1

    if usage_check:
        with open("cpu_usage.txt", "w") as c:
            c.write(cpu_usage_dump)
        with open("mem_usage.txt", "w") as m:
            m.write(mem_usage_dump)
    
    vid.release()
    cv2.destroyAllWindows()

    print("Result: {} frames".format(count))
    if write_output:
        record.close()
        trackedVideo.release()


