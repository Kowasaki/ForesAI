import cv2
import logging
import numpy as np
import time 

from utils.box_op import Box, parse_tf_output
from utils.fps import FPS
from utils.videostream import WebcamVideoStream
from utils.visualize import overlay

class InferenceBuilder:
    def __init__(self, config):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.is_camera = config["is_camera"]
        self.video_path = config["device_path"]
        self.task = config["task"]
        self.show_stream = config["show_stream"]
        self.write_output = config["write_output"]
        self.benchmark = config["benchmark"]
        self.ros_enabled = config["ros_enabled"]
        self.feed = self._set_video_feed(config["ros_enabled"])
        self.output = self._set_inferece_publisher(config["ros_enabled"])
        self.height, self.width = self.feed.get_dimensions()
        self.model = self._load_model(config["model"], self.height, self.width)
        self.graph_trace = False

        if config["model"]["graph_trace"] is not None:
            self. graph_trace = True

    def _set_video_feed(self, ros_enabled):
        if ros_enabled:
            from utils.ros_op import CameraSubscriber
            sub = CameraSubscriber() 
            if not sub.is_running():
                return Exception("[ERROR: Camera Node not running]")
            return sub   
        
        return WebcamVideoStream(src = self.video_path).start()


    def _set_inferece_publisher(self, ros_enabled):
        if ros_enabled:
            from utils.ros_op import DetectionPublisher
            return DetectionPublisher()
        #TODO: Add default publish method
        return None


    def _load_model(self, model_config, height, width):

        # Load the corresponding Loader for library
        if model_config["library"] == "tensorflow":
            from inference.loader.tf_model import TFModelLoader
            #TODO: Integrate the visualization tools
            from PIL import Image
            from tf_object_detection.utils import ops as utils_ops 
            from tf_object_detection.utils import visualization_utils as vis_util

            return TFModelLoader(model_config, height, width)
        elif model_config["library"] == "movidius":
            from inference.loader.mvnc_model import MovidiusModelLoader
            return MovidiusModelLoader(model_config, height, width)
        elif model_config["library"] == "pytorch":
            from inference.loader.pytorch_model import PyTorchModelLoader
            return PyTorchModelLoader(model_config, height, width)  
        else:
            raise Exception("[ERROR: Unsupported Library!]")

    def _visualize(self, task, output):
        # TODO: finish this; make visualization generalizable
        if task == "detect":
            pass
        elif task == "instance":
            pass
        elif task == "segmentation":
            from pytorch_segmentation.transform import Colorize
            # Visualizes based off of cityscape classes; this step takes a ton of time!
            label_color = Colorize()(ouput.unsqueeze(0))
            label_color = np.moveaxis(label_color.numpy(), 0, -1)
            label_color = label_color[...,::-1]

            if self.show_stream:
                window_name = "stream"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name,label_color)

            if self.write_output:
                self.trackedVideo.write(label_color)

    def detect(self):
        #TODO: make a generalized detection workflow
        labels_per_frame = []
        boxes_per_frame = []

        if self.benchmark:
            from benchmark.usage import Timer, UsageTracker
            self.logger.info("Initial startup")
            timer = Timer()
            usage = UsageTracker(timer)
            usage.get_usage()
        
        self.logger.debug("Frame width: {} height: {}".format(self.width, self.height))

        if self.write_output:
            self.trackedVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (c,r))
            self.record = open("record.txt", "w")

        count = 0

        if self.benchmark:
            fps = FPS().start()

        while self.feed.is_running:
            try:
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                self.logger.debug("Frame {}".format(count))
                retval, curr_frame = self.feed.read()

                if not retval:
                    self.logger.info("Video ending at frame {}".format(count))
                    break

                if self.show_stream:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                curr_frame_expanded = np.expand_dims(curr_frame, axis=0)

                # Actual detection.
                start = time.time()
                output = self.model.inference(curr_frame_expanded)            
                end = time.time()

                if self.benchmark:
                    fps.update()
                    self.logger.info("Session run time: {:.4f}".format(end - start))
                    self.logger.info("Frame {}".format(count))
                    usage.get_usage()
                
                # TODO: Publish Output
                if self.ros_enabled:
                    self.logger.info("Publishing via ROS")
                else:
                    self.logger.info("Publishing via custom module")
                
                if self.show_stream:
                    #TODO: set which type of visualization to use based on task 
                    self._visualize(self.task, output)
                
                count += 1

            except KeyboardInterrupt:
                self.logger.info("Ctrl + C Pressed. Attempting graceful exit")
                break
    
                        

                


    
