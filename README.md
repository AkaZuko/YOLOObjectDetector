# Object Detection using YOLO

## Introduction:
YOLO is a state-of-the-art object detection system. You can find more information about YOLO [here](https://pjreddie.com/darknet/yolo/)

**Disclaimer:** This is just an attempt to provide a utility that can perform object detection using YOLO.

After reading through multiple articles and checking out darkflow, I thought of providing a simple utility that can be used directly to perform Object Detection.

### Who can use this?
Anyone who is preparing ML pipeline and want to bootstrap object detection ensuring that targeted class is supported by the model already.

### How to set it up?
- Prepare a **python3** virtual env and activate the same
- Run: `bash setup.sh`

### How to validate the setup?
- `python src/utils/web_cam.py`

### Can I change the Tensor flow models being used?
Yes, you surely can. You can add your model under `models` directory and corresponding labels under `data` directory. Also, you can get the more models from [Tensor Flow models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
