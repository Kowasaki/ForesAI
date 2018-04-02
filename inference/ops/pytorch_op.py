import cv2
import numpy as np
import torch
import os
import importlib
import logging
import time

from benchmark.usage import Timer, get_cpu_usage, get_mem_usuage, print_cpu_usage, print_mem_usage, show_usage

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image

from pytorch_segmentation.transform import Colorize

from utils.box_op import Box, parse_tf_output
from utils.fps import FPS
from utils.videostream import WebcamVideoStream
from utils.visualize import overlay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    # for k in own_state:
    #     print(k)
    for name, param in state_dict.items():
        # if "module" not in name:
        #     name = "module.{}".format(name)
        if name not in own_state:
            logger.warning("element {} not found!".format(name))
            continue
        own_state[name].copy_(param)
    return model

def run_detection(video_path,
                  model_path,
                  model_name,
                  weights_path, 
                  classes,
                  show_window = True,
                  visualize = True, 
                  write_output = False,
                  is_cpu = False,
                  ros_enabled = False, 
                  usage_check = False):

    # Calling the class of the model 
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Net = getattr(mod, model_name)

    model = Net(classes)
    model = torch.nn.DataParallel(model)
    model = load_my_state_dict(model, torch.load(weights_path))
    model.eval()

    cpu_usage_dump = ""
    mem_usage_dump = ""
    time_usage_dump = ""

    if not is_cpu:
        model = model.cuda()
    else:
        return Exception("[ERROR: CPU mode not implemented]")
        
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

    logger.info("Video frame width: {} height: {}".format(r,c))

    if write_output:
        trackedVideo = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (c, r))
        record = open("record.txt", "w")

    count = 0

    if usage_check:
        fps = FPS().start()

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
            
            start = time.time()
            
            # Convert numpy img to PyTorch Tensor, then expand dimension for model
            convert = Compose([
                ToTensor()
            ])
            img_tensor = convert(curr_frame)
            img_tensor = img_tensor.unsqueeze(0)

            con = time.time()
            logger.debug("img conversion time: {:.4f}".format(con - start))

            if (not is_cpu):
                image = img_tensor.cuda()

            inputs = Variable(image, volatile=True)

            outputs = model(inputs)

            out = time.time()
            logger.debug("output time: {:.4f}".format(out - con))

            label = outputs[0].max(0)[1].byte().cpu().data # Mask to be published

            l = time.time()
            logger.debug("labeling time: {:.4f}".format(l - out))

            end = time.time()

            if usage_check:
                fps.update()
                logger.info("Session run time: {:.4f}".format(end - start))
                logger.info("Frame {}".format(count))
                cpu_usage_dump, mem_usage_dump, time_usage_dump  = show_usage(cpu_usage_dump, 
                    mem_usage_dump, time_usage_dump, timer)
            
            # TODO: Publish Segmentation
            if ros_enabled:
                logger.info("Publishing segmengatation via ROS")
            else:
                logger.info("Publishing segmentation via custom module")
            
            # Visualization of the results of a detection.
            if visualize:
                # Visualizes based off of cityscape classes; this step takes a ton of time!
                label_color = Colorize()(label.unsqueeze(0))
                label_color = np.moveaxis(label_color.numpy(), 0, -1)
                label_color = label_color[...,::-1]
                vis = time.time()
                logger.debug("visualization time: {:.4f}".format(vis - end))

                if show_window:
                    window_name = "stream"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name,label_color)

                if write_output:
                    trackedVideo.write(label_color)

            count += 1

            # # Quick benchmarking
            # if timer.get_elapsed_time() >= 60:
            #     break

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





