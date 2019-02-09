import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import glob

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

def proc_gt(val_labels):
    """Process ground truth labels so we only keep pedestrian labels
    Args:
    gt - groundtruth, taken selected labels from BDD val labels
    Returns:
    proc_gt - Processed groundtrugh labels, keeping only labels pertaining to pedestrians
    """
    gt = []
    for img_labels in val_labels:
        gt_label = {}
        gt_label['name'] = img_labels['name']
        gt_label['timeofday'] = img_labels['attributes']['timeofday']
        gt_label['weather'] = img_labels['attributes']['weather']
        gt_label['scene'] = img_labels['attributes']['scene']
        gt_xmin = []
        gt_ymin = []
        gt_xmax = []
        gt_ymax = []
        gt_classes = []
        gt_occluded = []
        person_labels = []
        for label in img_labels['labels']:
            if label['category'] == 'person':
                person_labels.append(label)
                gt_xmin.append(float(label['box2d']['x1']))
                gt_ymin.append(float(label['box2d']['y1']))
                gt_xmax.append(float(label['box2d']['x2']))
                gt_ymax.append(float(label['box2d']['y2']))
                gt_occluded.append(bool(label['attributes']['occluded']))
                gt_classes.append(1)

        gt_label['bbox_xmin'] = gt_xmin
        gt_label['bbox_ymin'] = gt_ymin
        gt_label['bbox_xmax'] = gt_xmax
        gt_label['bbox_ymax'] = gt_ymax
        gt_label['classes'] = gt_classes
        gt_label['occluded'] = gt_occluded
        gt.append(gt_label)
    return gt

def rescale_bbox(height, width, bounding_boxes):
    
    scaled_ymin = np.asarray(bounding_boxes[:,0] * height)
    scaled_xmin = np.asarray(bounding_boxes[:,1] * width)
    scaled_ymax = np.asarray(bounding_boxes[:,2] * height)
    scaled_xmax = np.asarray(bounding_boxes[:,3] * width)
    scaled_bbox = np.vstack((scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax)).T
    return scaled_bbox

def show_bbox(img, bbox, outputpath):
    for i, b in enumerate(bbox):
        ymin, xmin, ymax, xmax = b
        (height, width, channel) = img.shape
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
    cv2.imwrite(outputpath, img)

def display_gt_det_img(gt, inputdir, outputdir):
    for gt_img in gt:
        filename = gt_img['name']
        # open image
        img_path = inputdir + '/' + filename
        #print("input image path", img_path)
        img = cv2.imread(img_path)

        # get gt bbox from json file
        gt_img_path = outputdir + '/gt_' + filename

        # draw rectangle in image
        for i, classval in enumerate(gt_img['classes']):
            cv2.rectangle(img, (int(gt_img['bbox_xmin'][i]), int(gt_img['bbox_ymin'][i])), (int(gt_img['bbox_xmax'][i]), int(gt_img['bbox_ymax'][i])), (30, 255, 255), thickness=2)
        cv2.imwrite(gt_img_path, img)

        # display ground truth image
        # cv2's imshow will hang the notebook
        print("Groundtruth image", filename)
        dp_gt_img = dp.Image(filename=gt_img_path, format='jpg')
        dp.display(dp_gt_img)

        # display detected image
        print("Detection image", filename)
        detected_img_path = outputdir + '/det_' + filename
        dp_det_img = dp.Image(filename=detected_img_path, format='jpg')
        dp.display(dp_det_img)

def get_ap(recalls, precisions):
    """From BDD_data github """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap

def get_confusion(gt,predictions, threshold):
    detected_boxes = predictions['scaled_detection_boxes']
    detected_scores = predictions['detection_scores']
    gt_xmax = np.asarray(gt['bbox_xmax'])
    gt_xmin = np.asarray(gt['bbox_xmin'])
    gt_ymax = np.asarray(gt['bbox_ymax'])
    gt_ymin = np.asarray(gt['bbox_ymin'])
    
    tp = np.zeros(predictions['num_detections'])
    fp = np.zeros(predictions['num_detections'])
    gt_checked = np.zeros(len(gt['classes']))

    # scores are already sorted
    for i, score in enumerate(detected_scores):
        # if score is 0, module is not making any predictions
        if score == 0:
            continue
        ovmax = -np.inf
        jmax = -1

        if len(gt['classes']) > 0:

            detected_xmin = detected_boxes[i][1]
            detected_ymin = detected_boxes[i][0]
            detected_xmax = detected_boxes[i][3]
            detected_ymax = detected_boxes[i][2]

            # computer overlaps
            # intersection
            ixmin = np.maximum(gt['bbox_xmin'], detected_xmin)
            iymin = np.maximum(gt['bbox_ymin'], detected_ymin)
            ixmax = np.minimum(gt['bbox_xmax'], detected_xmax)
            iymax = np.minimum(gt['bbox_ymax'], detected_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            #union
            uni = ((detected_ymax - detected_ymin + 1.) *  (detected_xmax - detected_xmin + 1.) +
                   (gt_xmax - gt_xmin + 1.) *  (gt_ymax - gt_ymin + 1.) - inters)

            #print(detected_xmin, gt['bbox_xmin'], inters, uni)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        # non max suppression
        if ovmax > threshold:
            if gt_checked[jmax] == 0:
                tp[i] = 1
                gt_checked[jmax] = 1
            else:
                fp[i] = 1

    # Compute precision recall
    # fpcumsum = np.cumsum(fp, axis = 0)
    # tpcumsum = np.cumsum(tp, axis = 0)
    # num_gts = len(gt['classes'])
    
    # recalls = tpcumsum / num_gts
    # precisions = tpcumsum / np.maximum(tpcumsum + fpcumsum, np.finfo(np.float64).eps)

    # ap = get_ap(recalls, precisions)
    # return recalls[-1], tpcumsum[-1], num_gts

    num_gts = len(gt['classes'])
    tpsum = np.sum(tp)

    return tpsum/num_gts, tpsum, num_gts


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
        
def main():
    #gt = proc_gt('val_labels.json')
    input_image_path = 'inputdata/b1ca2e5d-84cf9134.jpg'
    image = Image.open(input_image_path)
    image_np = load_image_into_numpy_array(image)
    print(image_np.shape)
    
if __name__ == '__main__':
    main()
    
