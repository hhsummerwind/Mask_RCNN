#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[2]:


import os
import sys
import random
import math
import copy
import pdb
import numpy as np
import skimage.io
from skimage import draw
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

matplotlib.use('Agg')

ROOT_DIR = '/projects/open_sources/segmentation/Mask_RCNN'

cascade = False

if cascade:
    # Root directory of the project
    LOG_DIR = '/data/models/segment/cascade_mask_rcnn'
    vis_dir = '/data/models/segment/cascade_mask_rcnn/logs/vis'

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model_cascade as modellib
    # from mrcnn import visualize
else:
    # Root directory of the project
    LOG_DIR = '/data/models/segment/mask_rcnn'
    # vis_dir = '/data/models/segment/mask_rcnn/logs/vis
    vis_dir = '/data/datasets/tianji/heyue/result_khyh_seg'

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    # from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/text/"))  # To find local version
# from samples.coco import coco
from samples.text import train

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(LOG_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_text_0160.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(MODEL_PATH):
#     utils.download_trained_weights(MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = '/data/datasets/tianji/heyue/khyh'


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[3]:


class InferenceConfig(train.TextConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights

# In[4]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)

# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[5]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'text']


# ## Run Object Detection

# In[8]:

def find_largest_contour(contours):
    max_ind = 0
    max_area = 0
    for ind, c in enumerate(contours):
        if cv2.contourArea(c) > max_area:
            max_area = cv2.contourArea(c)
            max_ind = ind
    return max_ind


def is_useful_contour(contour, bbox, iou_threshold=0.6):
    (x, y, w, h) = cv2.boundingRect(contour)
    bbox_2 = [x, y, x + w, y + h]
    if calIoU(bbox, bbox_2) > iou_threshold:
        return True
    else:
        return False


def calIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1)
    garea = (gx2 - gx1) * (gy2 - gy1)

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h

    iou = float(area) / (carea + garea - area)
    return iou


def generate_polygon(mask, box):
    y1, x1, y2, x2 = box
    mask_int = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_int, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    r_box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if len(contours) > 0:
        max_ind = find_largest_contour(contours)
        contour = contours[max_ind]
        # print(contour)
        useful_box = [x1, y1, x2, y2]
        if is_useful_contour(contour, useful_box):
            rect = cv2.minAreaRect(contours[max_ind])
            r_box = np.int0(cv2.boxPoints(rect))
            r_box = np.reshape(r_box, [-1, 2])

            polygon = np.int0(contour)
            polygon = np.reshape(polygon, [-1, 2])

    return r_box, polygon


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox


# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
from samples.text.get_box import get_crop

for img_path in glob.glob(os.path.join(IMAGE_DIR, '*', '*')):
    # if not os.path.basename(img_path) == 'test2_crop.png':
    #     continue
    image = cv2.imread(img_path)[:, :, ::-1]
    print(img_path, image.shape)
    # pdb.set_trace()

    # Run detection
    results = model.detect([image], verbose=1)

    # print(results)

    # Visualize results
    r = results[0]
    # masked_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                                            class_names, r['scores'], show_mask=True)
    # plt.imsave(os.path.join(vis_dir, os.path.basename(img_path)), masked_image)
    # In[ ]:
    polygons = []
    r_boxes = []
    vis_image = copy.deepcopy(image)
    for i in range(len(r['rois'])):
        # for box, mask in zip(r['rois'], r['masks']):
        box = r['rois'][i]
        mask = r['masks'][:, :, i]
        # pdb.set_trace()
        # print(r['masks'].sum(), mask.sum())
        r_box, polygon = generate_polygon(mask, box)
        # polygons.append(polygon)
        r_boxes.append(r_box)
        # mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        # pdb.set_trace()
        # masked_image = image * mask
        # masked_image = np.uint8(masked_image)
        points = np.asarray(polygon)
        points = np.reshape(points, [-1, 2])
        cv2.polylines(vis_image, np.int32([points]), 1, (0, 255, 0), 2)
    plt.imsave(
        os.path.join(vis_dir, '{}_{}'.format(os.path.basename(os.path.dirname(img_path)), os.path.basename(img_path))),
        vis_image)
    # dst_dir = os.path.join('/data/tmp/{}'.format(os.path.basename(os.path.splitext(img_path)[0])))
    # if not os.path.exists(dst_dir):
    #     os.mkdir(dst_dir)
    # get_crop(image[:, :, ::-1], r_boxes, dst_dir)
