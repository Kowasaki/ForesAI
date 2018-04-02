# ForesAI
Lightweight Computer Vision library for intertergrating your deep learning models with camera devices to perform inference tasks.

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

This is a very early work in progress. The project stems out of my own research on efficient CNNs so I'm adding features/debugging as needed. Please check [To-Dos](#to-dos) for some upcoming tasks. However, I am looking for feedback as I want to library to support other use cases as well. Feel free to open an issue or make a pull request as you see fit--I am looking for additional contributors as I continue to build upon the library. 

# Notice
ForesAI supports vision-related tasks such as object detection, sematic segmentation, and instance segmenatation based on the relevant models. These APIs assume you have prepared a pre-trained model. For my TensorFlow models, all training/evaluation is done via the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). I will provide the scripts I used for my own training under a different repo in the future, but YMMV as much of it depends on your own system configurations.

Orignally the library was meant to support TensorFlow only, but as you can see the scope has increased drastically as my own research demanded. I'm in the process of building a standard, library-agnostic inferface to make building new inference workflows much easier. As such, all the run_detection functions in the ops files will be depreciated in the future. Feel free to look at the **model_loader** module under **inference** to get a sense of how it is being done.

Currently, ForesAI works with following deep learning libraries:

## Object Detection / Instance Segmentation
- TensorFlow models that are trained using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 

## Object Detection
- [Movidius NCS](https://github.com/movidius/ncsdk/) compiled models for object detection. 

## Semantic Segmentation
- [ERFNet](https://github.com/Eromera/erfnet_pytorch) 

I would like to support other libraries as well so let me know if you want to help in this endeaver!

# Requirements
Must haves:
- Python 3 (Python 2 might still work, but is untested)
    - Following packages:
        - numpy
        - psutil
- OpenCV3 (your own build or pip)

For TensorFlow:
- Pillow package
- [protobuf](https://github.com/google/protobuf)
- [TensorFlow](https://www.tensorflow.org/)

For Movidius:
- [Movidius SDK](https://movidius.github.io/ncsdk/)

For PyTorch:
- Pillow package
- [PyTorch](http://pytorch.org/)

# Instructions
If you don't have a model set up, feel free to use this [slightly modified SSD-mobilenetv1 model](https://drive.google.com/drive/folders/1Cwy89QCs3R2dFRxZ85TZJZFBFMtTsl0D?usp=sharing) here. You'll need both folders extracted within the "ForesAI" folder.

There are two main ways to access the module. If you want to run ForesAI as a standalone module:

```
python main.py --config_path <CONFIG_PATH> 
```
Where CONFIG_PATH is a json file with the configurations shown in demo_configs folder. If you want to test this out on your laptop **webcam_benchmark.json** would be a good first choice. Adding the "--benchmark" flag will show graphs measuring cpu/ram usage over time. One thing to notice is that the **device_path** in the config does not have to be an actual camera--a recorded video will work as well!

If you wish to use ForesAI as a package, you can start by running the webcam_benchmark_demo.py from your webcam to see how to use the camera detection API. You can also try the video_demo to have the object inference run on a video file of your choosing. For other configurations, please take a look at the *_demo.py scripts along with the respective JSON config files for how to use your own camera hardware. If using your own model, you will need to tweak the config json within the "demo_configs" folder.

# Benchmarks
These are the best benchmarks I got based on averages over a 1-minute stream. It is **very** likely that all of these can be improved with specific model-based hacks. There's a lot of good work done with the SSD-Mobilenet [here](https://github.com/GustavZ/realtime_object_detection)

**Jetson TX2**

|             |Frames per Second| CPU % | Combined RAM (MB) |
|:-----------:|:---------------:|:-----:|:-----------------:|
|SSD-Mobilenet (TensorFlow)|17|todo|450|
|SSD-Mobilenet (Movidius)|30|todo|todo|
|Mask-RCNN|Not Happening|N/A|OOM|
|ERFnet|12.5|todo|2400|
|ResNet 18-8|todo|todo|todo|
|ResNet 34-8|todo|todo|todo|

**Nvidia GeForce GTX 1070; i7; 16 GB RAM**

|             |Frames per Second| GPU RAM (MB) | CPU % | RAM (MB) |
|:-----------:|:---------------:|:------------:|:-----:|:--------:|
|SSD-Mobilenet (TensorFlow)|55|todo|todo|450|
|SSD-Mobilenet (Movidius)|Not tested|Not tested|Not tested|Not tested|
|Mask-RCNN|1.7|100%|todo|16|
|ERFnet|todo|todo|todo|2400|
|ResNet 18-8|todo|todo|todo|todo|
|ResNet 34-8|todo|todo|todo|todo|



# To-Dos
Right now I will only focus on features I need for my project in the immediate future, but I would love to hear from you about how to make this library useful in your own workflow!

- Documentation
- Make framework generalizable for custom models in Tensorflow and PyTorch (Model loaders)
- Interface for sending detections (e.g. a Publisher independent of ROS)
- Allow the user to implement manual, model-specific hacks 
- multi-stick support for movidus
- Add object tracker
- ROS integration
- Document functions
- Nvidia Tegra GPU usage monitoring (If on Jetson platform, you can just use tegrastats.sh)
- Nvidia NVML GPU usage monitoring (can also just use nividia-smi)



# Citations
## Code
Many thanks to all the sources cited below. All borrowed code are also cited in the same files they appear:

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 

- Stream related classes: [imutils](https://github.com/jrosebr1/imutils) 

- Mobilenet-related hacks that greatly improved speed from [realtime_object_detection](https://github.com/GustavZ/realtime_object_detection)

- [ERFNet implementation and helper functions](https://github.com/Eromera/erfnet_pytorch)

## Models
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017

"Efficient ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017. [Best Student Paper Award], [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)





