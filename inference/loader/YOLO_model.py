import copy
import cv2

from darknet.darknet import detect, call_load_net, call_load_meta

from inference.loader.model_loader import ModelLoader


class YOLOModelLoader(ModelLoader):
    def __init__(self, model_config, height, width):

        self.net = call_load_net(model_config["model_path"], model_config["weights_path"])
        self.meta = call_load_meta(model_config["label_path"])

    
    def inference(self, input):
        res = detect(self.net, self.meta, input)
        return res