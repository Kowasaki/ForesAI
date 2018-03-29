import logging

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
        height, width = self.feed.get_dimensions()
        self.model = self.load_model(config["model"], height, width)

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
        pass
        # while self.feed.is_running:


    
