import os
import sys
import cv2
import logging

top = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(top, os.path.pardir))
from detector import YOLODetector

from object_detection.utils import visualization_utils as vis_util

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

LOGGER = logging.getLogger(__file__)
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_FROZEN_GRAPH = os.path.join(os.path.pardir, os.path.pardir, 'models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(os.path.pardir, os.path.pardir, 'data', 'mscoco_label_map.pbtxt')

SHOW_DETECTION = True

def run():
  config = {
    "PATH_TO_FROZEN_GRAPH": PATH_TO_FROZEN_GRAPH,
    "PATH_TO_LABELS": PATH_TO_LABELS,
    "THRESHOLD": 0.7
  }
  cap = cv2.VideoCapture(0)
  _detector = YOLODetector(config)

  while True:
    _, frame = cap.read()
    predictions = _detector.detect(frame)
    LOGGER.info(predictions)

    if SHOW_DETECTION and predictions['num_detections'] > 0:
      vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        predictions['detection_boxes'],
        predictions['detection_classes'],
        predictions['detection_scores'],
        _detector.category_index,
        instance_masks=predictions.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=5)

      cv2.imshow('image', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == '__main__':
  run()
