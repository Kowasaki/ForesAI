import copy
import cv2
import logging
import numpy as np
import os
import signal
import tensorflow as tf
import time

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage, show_usage
from inference.loader.model_loader import ModelLoader
from inference.ops.tf_op import load_label_map, load_model, load_split_model
from tensorflow.core.framework import graph_pb2
from tf_object_detection.utils import label_map_util
from tf_object_detection.utils import ops as utils_ops 
from utils.box_op import Box, parse_tf_output
from utils.fps import FPS
# from utils.videostream import WebcamVideoStream
from utils.visualize import overlay

#TODO: Finish TFModelLoader
class TFModelLoader(ModelLoader):
    def __init__(self, model_config, height, width):

        PATH_TO_CKPT = model_config["model_path"]
        NUM_CLASSES = model_config["classes"]
        pbtxt = model_config["pbtxt"]
        split_model = model_config["split_hack"]
        
        if not split_model:
            self.graph = load_model(PATH_TO_CKPT)
        else:
            self.graph, self.score, self.expand = load_split_model(PATH_TO_CKPT)            
        self.label_map, self.categories, self.category_index = load_label_map(NUM_CLASSES, pbtxt)

        self.detection_graph = self.graph.as_default
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(graph=self.detection_graph, config = config)

        self.options = None
        self.run_metadata = None

        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            if 'detection_masks' in self.tensor_dict:
                detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, height, width)
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                self.tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Using the split model hack
        if self.score is not None and self.expand is not None:
            self.score_out = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            self.expand_out = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            self.score_in = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            self.expand_in = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')                
        
        if model_config["graph_trace"]:
            self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
            
    def inference(self, input):
        
        curr_frame_expanded = np.expand_dims(input, axis=0)

        if self.score is None and self.expand is None:
            output_dict = self.sess.run(self.tensor_dict,
                feed_dict={self.image_tensor: curr_frame_expanded},
                options=self.options,
                run_metadata=self.run_metadata)
        else:
            # Split Detection in two sessions.
            # TODO: implement the hack using the ModelLoader framework
            raise NotImplementedError
            # (score, expand) = self.sess.run(
            #     [self.score_out, self.expand_out], 
            #     feed_dict={self.image_tensor: curr_frame_expanded})
            # (boxes, scores, classes) = self.sess.run(
            #     [detection_boxes, detection_scores, detection_classes],
            #     feed_dict={score_in:score, expand_in: expand}) 

        # TODO: Just use the output_dict for visualization so these can be removed
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']
        classes =output_dict['detection_classes']

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0] 

        return output_dict, (boxes, scores, classes)


        


