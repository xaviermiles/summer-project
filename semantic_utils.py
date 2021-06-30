"""
Contains functions for computing metrics for segmantic segmentation (per-pixel classification).
All functions require the target and predicted annotations be in the form of single-integer (width x height) images (ie no channels/not RGB)
where the (non-negative) integer of a pixel corresponds to that pixels classification.
"""

import numpy as np
import skimage.transform
from sklearn import metrics
from tensorflow.keras.preprocessing.image import load_img


def convert_mrcnn_instances_to_semantic(mrcnn_result, config):
    """
    Given a result obtained from the Mask RCNN model via model.detect(), this 
    function outputs an image where the value of each pixel corresponds to
    the predicted class number (class location map).
    Requires: mrcnn_result
    Outputs: image with per-pixel classification encoded as integers
    ie. output is indexed by [width, height, class number]
    """
    masks = mrcnn_result['masks']
    does_pixel_have_mask = np.any(masks, axis=2)

    img_dim = list(config.IMAGE_SHAPE[:2])
    class_score_map = np.zeros(shape=tuple(img_dim + [config.NUM_CLASSES]))  # shape = (1024,1024,3)

    for instance_num, instance_class_id in enumerate(mrcnn_result['class_ids']):
        is_pixel_this_instance = masks[:, :, instance_num]

        # Construct 2D grid
        # where: pixel=score if the instance is there OR pixel=zero otherwise
        instance_score_map = np.zeros(shape=img_dim)
        instance_score_map[is_pixel_this_instance] = mrcnn_result['scores'][instance_num]

        # Merge current instance's instance_score_map with the existing score map
        # for that class (take maximum class score for each pixel)
        class_score_map[:, :, instance_class_id] = np.maximum(class_score_map[:, :, instance_class_id], instance_score_map)

    # For each pixel, take the class number with the highest score
    # Defaults to class 0 (background) if there is score=0 for the other classes
    out_int_img = np.argmax(class_score_map, axis=2)

    return out_int_img


def compute_metrics(target_img_paths, pred_imgs, num_classes, img_max_dim, display=False):
    """
    Computes mean and per-class metrics. Assumes square images
    Requires:
    - target_img_paths: list[string] with filepaths for all images with classes encoded as integers
    - pred_imgs: 3-D numpy array indexed by [image number, x, y]; the predicted images with classes encoded as integers
    - num_classes: (int)
    - img_max_dim: (int) will reshape imported images to square
    - display: (boolean) if True, print_results() will be called
    
    Returns dictionary which contains:
    > 'overall': 2-D numpy array indexed by [metric, image number]; weighted average of per-class metrics for each image using per-class support as weights
    > 'classwise': 3-D numpy array indexed by [metric, image number, class number]; per-class metrics for each image
    > 'conf_matrices': 3-D numpy array indexed by [image number, actual class, pred class]; confusion matrix for each image
    where the metrics are {IOU (Jaccard score), F1 score}
    """
    overall = np.zeros(shape=(2, len(target_img_paths)))
    classwise_loop = np.zeros(shape=(2, len(target_img_paths), num_classes))
    conf_matrices = np.zeros(shape=(len(target_img_paths), num_classes, num_classes),
                             dtype=np.uint64)

    labels = list(range(num_classes))  # classes = [0,1,...]

    for i, pred_int_img in enumerate(pred_imgs):
        target_int_img = np.array(load_img(target_img_paths[i]))
        # just need one channel since they contain identical information
        target_int_img_square = resize_image_to_square(target_int_img, max_dim=img_max_dim)[:, :, 1]
        
        target_vec = target_int_img_square.flatten()
        pred_vec = pred_int_img.flatten()
        
        # Calculate overall metrics (weighted avg of class metrics) for the i^th image
        overall[0, i] = metrics.jaccard_score(target_vec, pred_vec,
                                              labels=labels,
                                              average='weighted',
                                              zero_division=0)
        overall[1, i] = metrics.f1_score(target_vec, pred_vec,
                                         labels=labels,
                                         average='weighted',
                                         zero_division=0)
        
        # Calculate confusion matrix for the i^th image
        conf_matrices[i, :, :] = metrics.confusion_matrix(target_vec, pred_vec,
                                                          labels=labels)
    
    # Calculate classwise/per-class metrics for all images
    classwise = compute_all_classwise_metrics(conf_matrices)
     
    results = {
        'overall': overall, 
        'classwise': classwise,
        'conf_matrices': conf_matrices
    }
         
    if display:
        print_metric_results(results, num_classes)
        
    # Manually try to derive overall metrics
    num_pixels = np.sum(conf_matrices)
    

    return results


def compute_classwise_metrics(conf_matr):
    """
    Computes per-class IOU (Jaccard) score for a single image
    
    Requires:
    - conf_matr: square 2-D numpy array which contains actual (rows) and predicted (cols) classes
    
    Returns:
    2-D numpy array indexed by [metric class number]; per-class metrics
    where the metrics are {IOU (Jaccard score), F1 score}
    """
    num_classes = 3
    class_ious = np.zeros(shape=num_classes)
    class_F1s = np.zeros(shape=num_classes)
    
    for class_num in range(num_classes):
        intersection = conf_matr[class_num, class_num]
        union = np.sum(conf_matr[class_num, :]) + np.sum(conf_matr[:, class_num]) - intersection
        
        if union == 0:
            class_ious[class_num] = np.nan
            class_F1s[class_num] = np.nan
        else:
            class_ious[class_num] = intersection / union
            class_F1s[class_num] = 2 * intersection / (union + intersection)
        
    return np.stack([class_ious, class_F1s])


def compute_all_classwise_metrics(conf_matrices):
    """
    Compute per-class IOU (Jaccard) and F1 scores for a set of images
    
    Requires:
    - conf_matrices: 3-D numpy array indexed by [image_num, actual, pred]; set of square confusion matrices
    
    Returns:
    3-D numpy array indexed by [metric, image number, class number]; per-class metrics for each image
    where the metrics are {IOU (Jaccard score), F1 score}
    """
    num_images = conf_matrices.shape[0]
    num_classes = conf_matrices.shape[1]
    all_ious = np.zeros(shape=(num_images, num_classes))
    all_F1s = np.zeros(shape=(num_images, num_classes)) 
    
    for class_num in range(num_classes):
        intersections = conf_matrices[:, class_num, class_num]
        unions = np.sum(conf_matrices[:, class_num, :], axis=1) + np.sum(conf_matrices[:, :, class_num], axis=1) - intersections
        
        # IOU and F1 are undefined when union is zero
        with np.errstate(divide='ignore'):
            all_ious[:, class_num] = np.where(unions != 0, 
                                              intersections / unions, 
                                              np.nan)
            all_F1s[:, class_num] = np.where(unions != 0, 
                                             2 * intersections / (unions + intersections), 
                                             np.nan)
    
    return np.stack([all_ious, all_F1s], axis=0)
    
    
def print_metric_results(results, num_classes):
    """
    Prints out overall metrics, per-class metrics, and overall confusion matrix.
    Uses np.nanmean() for classwise results, as np.nan entries are used to indicate where an image does not have
    any actual or predicted pixels for a class (which means that IOU and F1 are undefined).
    
    Requires:
    - results: dictionary formatted as per the output of compute_metrics().
    
    Returns:
    None
    """
    # Average across images
    avg_overall = np.mean(results['overall'], axis=1)
    avg_classwise = np.nanmean(results['classwise'], axis=1)  # use nanmean to avoid np.nan entries
    avg_conf_matr = np.sum(results['conf_matrices'], axis=0)
    
    print("All classes  |   mIOU: {:.4f}   mF1 score: {:.4f}".format(
        avg_overall[0], avg_overall[1]))
    print(13 * " " + "|")

    for class_num in range(num_classes):
        print("Class {}      |    IOU: {:.4f}    F1 score: {:.4f}".format(
            class_num, avg_classwise[0, class_num], avg_classwise[1, class_num]))
    
    print()
    print("Overall confusion matrix:")
    print(avg_conf_matr)
    print()
    print("(rows=actual, cols=pred)")

    
def convert_to_rgb(int_mask, converter):
    """
    Converts single-integer Image (int_mask) to RGB Image.
    
    Requires:
    - int-mask: Numpy array
    - converter: dictionary with positive-integer keys (class) and 3-tuple values (colour); describes which
    colours to represent each class with
    """
    w, h, *_ = int_mask.shape
    colour_mask = np.zeros((w, h, 3), dtype=np.uint8)
    
    for class_num, colour in converter.items():
        modify_pixel = np.equal(int_mask, class_num)
        colour_mask[modify_pixel, :] = colour
    
    return colour_mask


def resize_image_to_square(image, min_dim=None, max_dim=None):
    """
    Resizes image to square according to min_dim/max_dim
    (Adapted from mrcnn.utils.resize_img())
    """
     # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    # Scale up?
    if min_dim:
        scale = max(1, min_dim / min(h, w))

    # Scale down?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        scaled_image = skimage.transform.resize(
            image=image, 
            output_shape=(round(h * scale), round(w * scale)),
            order=1, mode='constant', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None
        )
    
    # Get new height and width
    h, w = scaled_image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    square_image = np.pad(scaled_image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    
    return square_image.astype(image_dtype)
    