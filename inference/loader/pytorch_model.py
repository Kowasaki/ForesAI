import copy
import cv2
import importlib
import logging
import numpy as np
import os
import signal
import time
import torch

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage, show_usage
from inference.loader.model_loader import ModelLoader
from inference.ops.pytorch_op import load_my_state_dict

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image

from pytorch_segmentation.transform import Colorize


class PyTorchModelLoader(ModelLoader):
    def __init__(self, model_config, height, width):

        # Calling the class of the model 
        spec = importlib.util.spec_from_file_location(model_config["model_name"], 
            model_config["model_path"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        Net = getattr(mod, model_config["model_name"])

        self.model = Net(model_config["classes"])
        self.model = torch.nn.DataParallel(self.model)
        self.model = load_my_state_dict(self.model, torch.load(model_config["weights_path"]))
        self.model.eval()

        self.convert = Compose([
            ToTensor()
        ])

    
    def inference(self, input):
        
        img_tensor = self.convert(input)
        img_tensor = img_tensor.unsqueeze(0)
        image = img_tensor.cuda()
        input_var = Variable(image, volatile=True)
        outputs = self.model(input_var)
        # Mask to be published: GPU: add .data CPU add .cpu().data
        label_mask = outputs[0].max(0)[1].byte()
        return label_mask