import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import glob
from collections import defaultdict

# Read original BDD val labels and only select labels with pedestrians
def format_labels(images):
    format_labels = []
    for img in images:
        
        for l in img['labels']:
            if l['category'] != 'person':
                continue
            else:
                item = {}
                item['name'] = img['name']
                item['timeofday'] = img['attributes']['timeofday']
                item['occluded']  = l['attributes']['occluded']
                item['truncated'] = l['attributes']['truncated']
                item['bbox'] = [l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2']]
                
                format_labels.append(item)
    return format_labels

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def load_frozen_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
            
def load_image_into_numpy_array(image):
    (width, height) = image.size
    return np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    """ Mainly based off of tensorboard's run_inference_for_single_image code"""
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def show_bbox(img, bbox, outputpath):
    for i, b in enumerate(bbox):
        ymin, xmin, ymax, xmax = b
        (height, width, channel) = img.shape
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
    cv2.imwrite(outputpath, img)

def rescale_bbox(height, width, bounding_boxes):
    
    scaled_ymin = np.asarray(bounding_boxes[:,0] * height)
    scaled_xmin = np.asarray(bounding_boxes[:,1] * width)
    scaled_ymax = np.asarray(bounding_boxes[:,2] * height)
    scaled_xmax = np.asarray(bounding_boxes[:,3] * width)
    scaled_bbox = np.vstack((scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax)).T
    return scaled_bbox


def run_inference_show_bbox(frozen_graph_path, inputdir, outputdir, model_prefix):
    """ Run inference on all images in inputdata using provided frozen graph. copy images and draw bounding boxes into outputdir
    Args
       frozen_graph_path - frozen model
       inputdata - list of paths to input images
       model_prefix - prefix of the frozen model
    Returns:
       detection_result - detected class and bounding box
    """
    # load the frozen model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        
    detection_result = []
    inputdata = glob.glob(inputdir + '/*jpg')
    
    for input_image_path in inputdata:
        filename = input_image_path.replace(inputdir + '/', '')

        # open image
        image = Image.open(input_image_path)

        # prepare images to feed into the model 
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        output_dict['name'] = filename

        # rescale pixels to full range of width and height
        img = cv2.imread(input_image_path)
        height, width, channel = img.shape
        output_dict['scaled_detection_boxes'] = rescale_bbox(height, width, output_dict['detection_boxes'])

        # Draw detected bounding boxes and store in outputdir
        det_image_path = outputdir + '/det_' + model_prefix + '_' + filename
        show_bbox(img, output_dict['scaled_detection_boxes'], det_image_path)
        detection_result.append(output_dict)
        print(filename, "Number of detected pedestrians", output_dict['num_detections'])
    return detection_result

# format the prediction results for later use
def format_pred(predictions):
    format_pred = []
    for img in predictions:
        
        for idx in range(int(img['num_detections'])):
            item = {}
            item['name'] = img['name']
            item['bbox']  = [img['scaled_detection_boxes'][idx][1], img['scaled_detection_boxes'][idx][0],
                            img['scaled_detection_boxes'][idx][3], img['scaled_detection_boxes'][idx][2]]
            
            item['score'] = img['detection_scores'][idx]

            format_pred.append(item)
    return format_pred
