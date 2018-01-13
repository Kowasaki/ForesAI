# ForesAI
Lightweight Computer Vision library for using TensorFlow models to perform inference tasks

# Notice
Currently, All training/evaluation is done via the TensorFlow Object Detection API. I will provide the scripts I used for my own training under "training" in the very near future, but YMMV as much of it depends on your own system configurations.

I will be focused on removing the dependencies on the API for inference bit by bit as my own research project continues.

# Requirements
Python 2 or 3
TensorFlow
OpenCV (v3 or above)
Requirements from [TensorFlow Object Detection API] (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

# Training Instructions
Please refer to the detailed tutorial at https://github.com/tensorflow/models/tree/master/research/object_detection for how to use the Object Detection API for training/evaluation

## Do this before running the api or save it into bashrc:
### From tensorflow/models/research/
#### For nix systems:
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#### For windows systems (also from research folder):
py -3 setup.py build
py -3 setup.py install






