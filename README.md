# ForesAI
Lightweight Computer Vision library for using TensorFlow models to perform inference tasks in embedded AI systems

# Introduction
Applications that utilizes machine learning models for vision tasks has been growing rapidly in recent years, and thus the need for tools that integrate between the data science and software engineering pipelines. ForesAI aims to be the bridge the gap between the two by providing a library with simple APIs for you to apply your machine learning models directly into your product across different platforms. With a particular emphasis on robotic use cases, ForesAI aims to minimize resource usage so that it can run on as many different hardware as possible and leave room for you to build out the rest of the AI system.

This is a very early work in progress but I am looking for feedback as I want to library to be useful for others as well. Feel free to open an issue or make a pull request as you see fit--I am looking for additional contributors as I continue to build upon the library. 

# Notice
These APIs assume you have prepared a pre-trained model. For my TensorFlow models, all training/evaluation is done via the TensorFlow Object Detection API. I will provide the scripts I used for my own training under "training" in the very near future, but YMMV as much of it depends on your own system configurations.

The "tf_object_detection" module contains code that comes directly from [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). I will be focused on removing the dependencies on the API for inference bit by bit as I continue to improve upon the efficiency of library

# Requirements
## Must haves:
- Python 2 or 3
    - Following packages:
        - numpy
        - psutil
- OpenCV3 (your own build or pip)

## For TensorFlow:
- [protobuf](https://github.com/google/protobuf)
- [TensorFlow](https://www.tensorflow.org/)

## For Movidius:
- [Movidius SDK](https://movidius.github.io/ncsdk/)


# Roadmap
Right now the list will only consist of things I need for my project in the immediate future, but I would love to hear from you about how to make this library useful in your own workflow!

- Interface for sending detections every frame
- multi-stick support for movidus
- Add object tracker
- ROS integration
- image segmentation tasks
- Remove dependencies from Object Detection API for inference
- Document functions
- Nvidia Tegra GPU usage monitoring (If on Jetson platform, you can just use tegrastats.sh)
- Nvidia NVML GPU usage monitoring (can also just use nividia-smi)

# Instructions
Currently, ForesAI works with following deep learning models:
- TensorFlow models that are trained using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) 

- [Movidius NCS](https://github.com/movidius/ncsdk/) compiled models for object detection. 

I would like to support other libraries as well so let me know if you want to help in this endeaver!

Please take a look at all the *_demo.py scripts for how to use your respective camera hardware. My suggestion is to start by running the webcam_demo from your laptop to see how to use the camera detection API. You can also try the video_demo to have the object inference run on a video file of your choosing.

## For Training (TensorFlow):
You will need the requirements from [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

Please refer to the detailed tutorial at https://github.com/tensorflow/models/tree/master/research/object_detection for how to use the Object Detection API for training/evaluation

## Do this before running the api or save it into bashrc:
### From tensorflow/models/research/
#### For nix systems:
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#### For windows systems (also from research folder):
py -3 setup.py build
py -3 setup.py install

# Citations
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017





