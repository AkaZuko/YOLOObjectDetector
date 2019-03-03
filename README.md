# Object Detection using YOLO

## Introduction:
YOLO is a state-of-the-art object detection system. You can find more information about YOLO [here](https://pjreddie.com/darknet/yolo/)

**Disclaimer:** This is just an attempt to provide a utility that can perform object detection using YOLO.

After reading through multiple articles and checking out darkflow, I thought of providing a simple utility that can be used directly to perform Object Detection.

### Who can consume this?
Anyone who is preparing ML pipeline and want to bootstrap object detection ensuring that targeted class is supported by the model already.

### How to set it up?

**Note:** Project uses python3

- Prepare a virtual env and activate the same
- Run: `bash setup.sh`

### How to validate the setup?
- `python src/utils/web_cam.py`