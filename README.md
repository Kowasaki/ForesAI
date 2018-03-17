# ForesAI
Lightweight Computer Vision library for using deep learning models to perform inference tasks in edge AI systems

# Introduction
Applications that utilizes machine learning models for vision tasks has been growing rapidly in recent years, and thus the need for tools that integrate between the data science and software engineering pipelines. ForesAI aims to be the bridge the gap between the two by providing a library with simple APIs for you to apply your machine learning models directly into your camera devices across different platforms. With a particular emphasis on robotic use cases, ForesAI aims to minimize resource usage so that you can run your models on as many different hardware as possible and provide you with the tools to broadcast your outputs to the rest of your AI system.

This is a very early work in progress. I am looking for feedback as I want to library to support other use cases as well. Feel free to open an issue or make a pull request as you see fit--I am looking for additional contributors as I continue to build upon the library. 

# Notice
ForesAI supports vision-related tasks such as object detection, sematic segmentation, and instance segmenatation based on the relevant models. These APIs assume you have prepared a pre-trained model. For my TensorFlow models, all training/evaluation is done via the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). I will provide the scripts I used for my own training under "training" in the very near future, but YMMV as much of it depends on your own system configurations.

Currently, ForesAI works with following deep learning models:

## Object Detection / Instance Segmentation
- TensorFlow models that are trained using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 

## Object Detection
- [Movidius NCS](https://github.com/movidius/ncsdk/) compiled models for object detection. 

## Semantic Segmentation
-[ERFNet](https://github.com/Eromera/erfnet_pytorch) 

I would like to support other libraries as well so let me know if you want to help in this endeaver!

# Requirements
## Must haves:
- Python 3 (Python 2 might still work, but is untested)
    - Following packages:
        - numpy
        - psutil
- OpenCV3 (your own build or pip)

## For TensorFlow:
- [protobuf](https://github.com/google/protobuf)
- [TensorFlow](https://www.tensorflow.org/)

## For Movidius:
- [Movidius SDK](https://movidius.github.io/ncsdk/)

## For PyTorch:
- [PyTorch](http://pytorch.org/)

# To-Dos
Right now the list will only consist of things I need for my project in the immediate future, but I would love to hear from you about how to make this library useful in your own workflow!

- Interface for sending detections (e.g. a Publisher independent of ROS)
- Manual, model-specific hacks
- multi-stick support for movidus
- Add object tracker
- ROS integration
- Document functions
- Nvidia Tegra GPU usage monitoring (If on Jetson platform, you can just use tegrastats.sh)
- Nvidia NVML GPU usage monitoring (can also just use nividia-smi)

# Instructions
If you don't have a model set up, feel free to use this [slightly modified SSD-mobilenetv1 model](https://drive.google.com/drive/folders/1Cwy89QCs3R2dFRxZ85TZJZFBFMtTsl0D?usp=sharing) here. You'll need both folders extracted within the "ForesAI" folder.

 My suggestion is to start by running the webcam_benchmark_demo.py from your webcam to see how to use the camera detection API. You can also try the video_demo to have the object inference run on a video file of your choosing. For other configurations, please take a look at the *_demo.py scripts along with the respective JSON config files for how to use your own camera hardware. If using your own model, you will need to tweak the config json within the "demo_configs" folder.

# Training Tips (TensorFlow):
You will need the requirements from [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

Please refer to the detailed tutorial at https://github.com/tensorflow/models/tree/master/research/object_detection for how to use the Object Detection API for training/evaluation

### Do this before running the api or save it into bashrc:
#### From tensorflow/models/research/
##### For nix systems:
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

##### For windows systems (also from research folder):
py -3 setup.py build
py -3 setup.py install

# Citations

[realtime_object_detection](https://github.com/GustavZ/realtime_object_detection)

"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017

"Efficient ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017. [Best Student Paper Award], [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), December 2017. [pdf](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)





