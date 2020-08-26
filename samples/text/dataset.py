# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: Mask_RCNN	
@file: dataset.py	
@version: v1.0
@time: 2020/5/14 下午5:02
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

import numpy as np
import os
import json
import cv2
from pycocotools import mask as maskUtils
import glob
import random
import copy

from mrcnn import utils


############################################################
#  Dataset
############################################################


class TextDataset(utils.Dataset):
    def __init__(self):
        super(TextDataset, self).__init__()
        self.source = 'text'

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def load_image(self, image_id):
        return super(TextDataset, self).load_image(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        # class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:

            m = self.annToMask(annotation, image_info["height"],
                               image_info["width"])
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            instance_masks.append(m)
        class_ids = [1, ] * len(instance_masks)
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(TextDataset, self).load_mask(image_id)

    def _get_annotations(self, labels, image_id):
        result_list = []
        for region in labels:
            transcription = region["transcription"]
            points = np.array(region["points"])
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            result_list.append({"segmentation": [[i for item in points for i in item]],
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [min_x, max_x, max_x - min_x, max_y - min_y],
                                'label': transcription},
                               )
        return result_list

    def __add__(self, other):
        assert isinstance(other, TextDataset)
        output = TextDataset()
        output.class_info = self.class_info
        output.image_info = self.image_info
        # output.class_info.append({'source': other.source, 'id': len(self.class_info), 'name': 'text'})
        for i, item in enumerate(other.image_info):
            new_dict = copy.deepcopy(item)
            new_dict['id'] = len(self.image_ids) + i
            output.image_info.append(new_dict)
        output.prepare()
        return output

    def __getitem__(self, item):
        output = TextDataset()
        output.class_info = self.class_info
        output.image_info = self.image_info[item]
        return output


class ArtTextDataset(TextDataset):
    def __init__(self):
        super(ArtTextDataset, self).__init__()
        # self.source = 'art'

    def _add_images(self, all_img_paths, annotations):
        random.shuffle(all_img_paths)
        self.add_class(self.source, 1, "text")

        for i, img_path in enumerate(all_img_paths):
            label_key = os.path.basename(os.path.splitext(img_path)[0])
            labels = annotations[label_key]
            img = cv2.imread(img_path)
            h, w, c = img.shape
            self.add_image(
                self.source, image_id=i,
                path=img_path,  # os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=w, height=h,
                annotations=self._get_annotations(labels, i))

    def load_text(self, data_dir):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        # data_dir = data_dir
        ann_file = os.path.join(data_dir, 'train_labels.json')
        annotations = json.load(open(ann_file, 'r'))
        all_img_paths = glob.glob(os.path.join(data_dir, 'train_images', '*'))[:10]
        self._add_images(all_img_paths, annotations)

    def __getitem__(self, item):
        output = ArtTextDataset()
        output.class_info = self.class_info
        output.image_info = self.image_info[item]
        return output


class RectsTextDataset(ArtTextDataset):
    def __init__(self):
        super(RectsTextDataset, self).__init__()
        # self.source = 'rects'

    def _get_annotations(self, labels, image_id):
        result_list = []
        lines = labels['lines']
        for region in lines:
            transcription = region["transcription"]
            points = np.array(region["points"]).reshape((-1, 2))
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            result_list.append({"segmentation": [region["points"]],
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [min_x, max_x, max_x - min_x, max_y - min_y],
                                'label': transcription},
                               )
        return result_list

    def _add_images(self, all_img_paths, annotations):
        random.shuffle(all_img_paths)
        self.add_class(self.source, 1, "text")

        for i, img_path in enumerate(all_img_paths):
            # label_key = os.path.basename(os.path.splitext(img_path)[0])
            # labels = annotations[label_key]
            label_path = os.path.join(annotations, os.path.basename(os.path.splitext(img_path)[0]) + '.json')
            labels = json.load(open(label_path, 'r'))
            img = cv2.imread(img_path)
            h, w, c = img.shape
            self.add_image(
                self.source, image_id=i,
                path=img_path,
                width=w, height=h,
                annotations=self._get_annotations(labels, i))

    def load_text(self, data_dir):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        # data_dir = data_dir
        ann_dir = os.path.join(data_dir, 'gt_unicode')
        # annotations = json.load(open(ann_file, 'r'))
        all_img_paths = glob.glob(os.path.join(data_dir, 'img', '*'))[:10]
        self._add_images(all_img_paths, ann_dir)


class LSVTTextDateset(ArtTextDataset):
    def __init__(self):
        super(LSVTTextDateset, self).__init__()
        # self.source = 'lsvt'

    def load_text(self, data_dir):
        all_img_paths = glob.glob(os.path.join(data_dir, 'train_full_images_*', '*'))[:10]
        ann_file = os.path.join(data_dir, 'train_full_labels.json')
        annotations = json.load(open(ann_file, 'r'))
        self._add_images(all_img_paths, annotations)


class RCTWTextDataset(RectsTextDataset):
    def __init__(self):
        super(RCTWTextDataset, self).__init__()
        # self.source = 'rctw'

    def _add_images(self, all_img_paths, annotations):
        random.shuffle(all_img_paths)
        self.add_class(self.source, 1, "text")

        for i, img_path in enumerate(all_img_paths):
            label_path = img_path.replace('.jpg', '.txt')
            # labels = json.load(open(label_path, 'r'))
            f = open(label_path, 'r')
            img = cv2.imread(img_path)
            h, w, c = img.shape
            self.add_image(
                self.source, image_id=i,
                path=img_path,
                width=w, height=h,
                annotations=self._get_annotations(f, i))
            f.close()

    def _get_annotations(self, labels, image_id):
        result_list = []
        # lines = labels['lines']
        for region in labels.readlines():
            contents = region.split(',')
            transcription = contents[-1][1:-2]
            try:
                points = np.array(contents[:-2], dtype=np.int).reshape((-1, 2))
            except:
                continue
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            result_list.append({"segmentation": [list(map(int, contents[:-2]))],
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [min_x, max_x, max_x - min_x, max_y - min_y],
                                'label': transcription},
                               )
        return result_list

    def load_text(self, data_dir):
        all_img_paths = glob.glob(os.path.join(data_dir, 'part*', '*.jpg'))[:10]
        self._add_images(all_img_paths, None)


if __name__ == '__main__':
    dataset1 = ArtTextDataset()
    dataset1.load_text('/data/datasets/text_recognition/ICDAR2019/art')
    dataset1.prepare()
    dataset2 = RectsTextDataset()
    dataset2.load_text('/data/datasets/text_recognition/ICDAR2019/rects')
    dataset2.prepare()
    output = dataset1 + dataset2
    dataset3 = LSVTTextDateset()
    dataset3.load_text('/data/datasets/text_recognition/ICDAR2019/lsvt')
    dataset3.prepare()
    dataset4 = RCTWTextDataset()
    dataset4.load_text('/data/datasets/text_recognition/ICDAR2017/rctw/icdar2017rctw_train_v1.2/train')
    dataset4.prepare()
    pass
