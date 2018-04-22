# ForesAI
Lightweight Computer Vision library for integrating your deep learning models with camera devices to perform inference tasks.

# Table of Contents
- [Introduction](#introduction)
- [Notice](#notice)
- [Requirements](#requirements)
- [Instructions](#instructions)
- [Benchmarks](#benchmarks)
- [To-Dos](#to-dos)
- [Citations](#citations)

# Introduction
Applications that utilizes machine learning models for vision tasks has been growing rapidly in recent years, and thus the need for tools that integrate between the data science and engineering pipelines. ForesAI aims to be the bridge the gap between the two by providing a library with simple APIs for you to apply your machine learning models built in popular libraries directly to your camera devices across different hardware platforms. With a particular emphasis on robotic use cases, ForesAI aims to minimize resource usage so that you can run your models on as many different hardware configurations as possible and provide you with the tools to broadcast your outputs to the rest of your AI system.

This is an early work in progress. The project stems out of my own research on efficient CNNs so I'm adding features/debugging as needed. Please check [To-Dos](#to-dos) for some upcoming tasks. However, I am looking for feedback as I want to library to support other use cases as well. Feel free to open an issue or make a pull request as you see fit--I am looking for additional contributors as I continue to build upon the library. 

# Notice
ForesAI supports vision-related tasks such as object detection, sematic segmentation, and instance segmenatation based on the relevant models. These APIs assume you have prepared a pre-trained model. For my TensorFlow models, all training/evaluation is done via the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). I will provide the scripts I used for my own training under a different repo in the future, but YMMV as much of it depends on your own configurations.

Orignally the library was meant to support TensorFlow only, but as you can see the scope has increased drastically as my own research demanded. I'm in the process of building a standard, library-agnostic inferface to make building new inference workflows much easier. As such, all the run_detection functions in the ops files will be depreciated in the future. Feel free to look at the **model_loader** module under **inference** to get a sense of how it is being done.

Currently, ForesAI supports the following tasks:

## Object Detection / Instance Segmentation
TensorFlow models that are trained using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Check out the [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) For the full list. Both SSD-Mobilenet and Mask-RCNN has been tested.


## Object Detection
[Movidius NCS](https://github.com/movidius/ncsdk/) compiled models for object detection. See the [NC App Zoo](https://github.com/movidius/ncappzoo) for details. The Caffe-compiled version of SSD-Mobilenet was tested.

## Semantic Segmentation
All [PyTorch](http://pytorch.org/) models. Right now visualization only works for the cityscape dataset. I've included [ERFNet](https://github.com/Eromera/erfnet_pytorch), [Resnet18 8s](https://github.com/warmspringwinds/pytorch-segmentation-detection), and [Resnet34 8s](https://github.com/warmspringwinds/pytorch-segmentation-detection) for use.

I would like to support additional libraries as well so let me know if you want to help in this endeaver!

# Requirements
Must haves:
- Python 3.5 or above (any other version will require minor fixes)
    - Following packages:
        - numpy
        - psutil
- OpenCV3 (your own build or pip)

For TensorFlow:
- Pillow (PIL) (For instance segmentation)
- [TensorFlow](https://www.tensorflow.org/)

For Movidius:
- [Movidius SDK](https://movidius.github.io/ncsdk/)

For PyTorch (Currently, only GPU mode w/CUDA is supported at this time):
- Pillow (PIL)
- [PyTorch](http://pytorch.org/)

# Instructions
If you don't have a model in mind, feel free to use this [slightly modified SSD-mobilenetv1 model](https://drive.google.com/drive/folders/1Cwy89QCs3R2dFRxZ85TZJZFBFMtTsl0D?usp=sharing) here to test out the object detection function. You'll need both folders extracted within the "ForesAI" folder.

There are two main ways to access the module. If you want to run ForesAI as a standalone module:

```
python main.py --config_path <CONFIG_PATH> 
```
Where CONFIG_PATH is a json file with the configurations shown in demo_configs folder. If you want to test this out on your laptop **webcam_benchmark.json** would be a good first choice. Adding the "--benchmark" flag will show graphs measuring cpu/ram usage over time. One thing to notice is that the **device_path** in the config does not have to be an actual camera--a recorded video will work as well!

If you wish to use ForesAI as a package, you can start by running the webcam_benchmark_demo.py from your webcam to see how to use the camera detection API. You can also try the video_demo to have the object inference run on a video file of your choosing. For other configurations, please take a look at the *_demo.py scripts along with the respective JSON config files for how to use your own camera hardware. If using your own model, you will need to tweak the config json within the "demo_configs" folder.

# Benchmarks
These are the best benchmarks I got based on averages over a 1-minute stream. The precision benchmarks come from reports by their specific authors. It is **very** likely that all of these can be improved with specific model-based hacks. There's a lot of good work done with the SSD-Mobilenet [here](https://github.com/GustavZ/realtime_object_detection).

**Jetson TX2; jetson_clocks enabled; Resolution 480x480**

|Object Detection Models|Frames per Second| CPU % | Combined RAM (MB) | COCO mAP |
|:---------------------:|:---------------:|:-----:|:-----------------:|:--------:|
|SSD-Mobilenet v1 (TensorFlow)|10.01|64.38|1838|21|
|SSD-Mobilenet v1 (TensorFlow, GPU/CPU Split)|18.02|54.89|1799|21|
|SSD-Mobilenet v1 (Movidius)*|10.08|10|247|Not Reported|

|Segmentation Models|Frames per Second| CPU % | Combined RAM (MB) | Mean IoU |
|:-----------------:|:---------------:|:-----:|:-----------------:|:--------:|
|ERFnet|7.54|23.54|2464|69.8|
|ResNet 18-8**|3.40|13.89|2297|N/A|
|ResNet 34-8**|1.85|13.26|2296|N/A|

*Measurement less accurate due to not using system tools instead of benchmarking module

**Both ResNet 18 and Resnet 34 requires changing the upsampling algorithm from bilinear interpolation to nearest neighbor for the models to run on the TX2, which will have a negative impact original reported mean IOU 

**Nvidia GeForce GTX 1070 8GB GDDR5; i7 4-core 4.20 GHz; 16 GB RAM; Resolution 480x480**

|Object Detection Models|Frames per Second| GPU RAM (MB) | CPU % | RAM (MB) | COCO mAP |
|:---------------------:|:---------------:|:------------:|:-----:|:--------:|:--------:|
|SSD-Mobilenet v1 (TensorFlow)|32.17|363|40.25|1612|21|
|SSD-Mobilenet v1 (TensorFlow, GPU/CPU Split)|61.97|363|58.09|1612|21|
|SSD-Mobilenet v1 (Movidius)|8.51|0|6|57|Not Reported|
|SSD-Mobilenet v2 (TensorFlow)|53.96|2491|35.94|1838|22|
|Mask-RCNN Inception v2|15.86|6573|22.54|1950|25|

|Segmentation Models|Frames per Second| GPU RAM (MB) | CPU % | RAM (MB) | Mean IoU |
|:-----------------:|:---------------:|:------------:|:-----:|:--------:|:--------:|
|ERFnet*|63.38|549|40.01|2181|69.8|
|ResNet 18-8*|38.85|605|31.07|2023|60.0|
|ResNet 34-8*|21.12|713|23.29|2020|69.1|

*For some reason running python in the virtual environment for TensorFlow decreased CPU usage by 20%(!). Need to figure out why...

**Measurement less accurate due to not using system tools instead of benchmarking module

# To-Dos
Right now I will only focus on features I need for my project in the immediate future, but I would love to hear from you about how to make this library useful in your own workflow!

- Documentation
- Make framework generalizable for custom models in Tensorflow and PyTorch (Model loaders)
- Interface for sending detections (e.g. a Publisher independent of ROS)
- Allow the user to implement manual, model-specific hacks 
- Standardizing visualization for each task
- multi-stick support for movidus
- Add object tracker
- ROS integration
- Nvidia Tegra GPU usage monitoring (If on Jetson platform, you can just use tegrastats.sh)
- Nvidia NVML GPU usage monitoring (can also just use nividia-smi)



# Citations
## Code
Many thanks to all the sources cited below. Please feel free to contact me if you have any questions/concerns:

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 

- Stream related classes: [imutils](https://github.com/jrosebr1/imutils) 

- Mobilenet-related hacks that greatly improved speed from [realtime_object_detection](https://github.com/GustavZ/realtime_object_detection)

- [ERFNet implementation and helper functions](https://github.com/Eromera/erfnet_pytorch)

- [Image Segmentation and Object Detection in Pytorch](https://github.com/warmspringwinds/pytorch-segmentation-detection)

## Models
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017

"Efficient ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017. [Best Student Paper Award], [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

@article{pakhomov2017deep,
  title={Deep Residual Learning for Instrument Segmentation in Robotic Surgery},
  author={Pakhomov, Daniil and Premachandran, Vittal and Allan, Max and Azizian, Mahdi and Navab, Nassir},
  journal={arXiv preprint arXiv:1703.08580},
  year={2017}
}




