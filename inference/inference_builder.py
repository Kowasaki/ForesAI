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
        self.show_stream = config["show_stream"]
        self.write_output = config["write_output"]
        self.benchmark = config["benchmark"]
        self.feed = self.set_video_feed(config["ros_enabled"])
        self.output = self.set_inferece_publisher(config["ros_enabled"])
        self.height, self.width = self.feed.get_dimensions()
        self.model = self.load_model(config["model"], self.height, self.width)
        self.graph_trace = False

        if config["model"]["graph_trace"] is not None:
            self. graph_trace = True

    def set_video_feed(self, ros_enabled):
        if ros_enabled:
            from utils.ros_op import CameraSubscriber
            sub = CameraSubscriber() 
            if not sub.is_running():
                return Exception("[ERROR: Camera Node not running]")
            return sub   
        
        return WebcamVideoStream(src = self.video_path).start()


    def set_inferece_publisher(self, ros_enabled):
        if ros_enabled:
            from utils.ros_op import DetectionPublisher
            return DetectionPublisher()
        #TODO: Add default publish method
        return None


    def load_model(self, model_config, height, width):

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
            trackedVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (c,r))
            record = open("record.txt", "w")

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
                self.model.inference(curr_frame_expanded)            
                end = time.time()

                if self.benchmark:
                    fps.update()
                    self.logger.info("Session run time: {:.4f}".format(end - start))
                    self.logger.info("Frame {}".format(count))
                    usage.get_usage()
            

            except KeyboardInterrupt:
                self.logger.info("Ctrl + C Pressed. Attempting graceful exit")
                break
    
                        

                


    
