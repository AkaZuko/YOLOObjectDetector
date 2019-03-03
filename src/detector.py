import logging
import tensorflow as tf
import numpy as np

from datetime import datetime

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
LOGGER = logging.getLogger(__file__)

class YOLODetector(object):

  def __init__(self, config):
    self.config = config
    self.graph = None
    self.session = None
    self.tensor_dict = None
    self.image_tensor = None

    self.__load_detector()

  def detect(self, image):
    threshold = self.config.get('THRESHOLD', 0.5)
    max_val = -1
    _predictions = {
      'num_detections': 0,
      'detection_classes': [],
      'detection_class_labels': [],
      'detection_boxes': [],
      'detection_scores': []
    }

    start_time = datetime.utcnow()
    predictions = self.__run_inference_for_single_image(image)
    delta = datetime.utcnow() - start_time
    LOGGER.info("Time taken to predict : {0}".format(delta))
    
    for idx in range(len(predictions['detection_scores'])):
      if predictions['detection_scores'][idx] > max_val:
        max_val = predictions['detection_scores'][idx]

      if predictions['detection_scores'][idx] >= threshold:
        _class = self.category_index[predictions['detection_classes'][idx]]['name']

        _predictions['num_detections'] += 1
        _predictions['detection_class_labels'].append(_class)
        _predictions['detection_boxes'].append(predictions['detection_boxes'][idx])
        _predictions['detection_scores'].append(predictions['detection_scores'][idx])
        _predictions['detection_classes'].append(predictions['detection_classes'][idx])

    del predictions
    LOGGER.info('Max detection score found this iteration: {0}'.format(max_val))

    if max_val > threshold:
      for key in _predictions:
        _predictions[key] = np.array(_predictions[key])  
    
    LOGGER.info(_predictions)
    return _predictions

  def __run_inference_for_single_image(self, image):
    # Run inference
    output_dict = self.session.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict

  def __load_detector(self):
    path_to_frozen_graph = self.config['PATH_TO_FROZEN_GRAPH']
    path_to_labels = self.config['PATH_TO_LABELS']
    tensor_dict = {}
    image_tensor = None
    sess = None

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

      sess = tf.Session()

      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      
      for key in [
          'num_detections', 
          'detection_boxes', 
          'detection_scores',
          'detection_classes'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    self.session = sess
    self.tensor_dict = tensor_dict
    self.image_tensor = image_tensor
    self.category_index = category_index
