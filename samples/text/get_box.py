# -*- coding: utf-8 -*-
"""
Copyright 2020 huhui, Inc. All Rights Reserved.
@author: huhui
@software: PyCharm	
@project: Mask_RCNN	
@file: get_box.py	
@version: v1.0
@time: 2020/5/22 下午3:36
@setting: 
-------------------------------------------------
Description :
工程文件说明： 
"""

from numpy import array
import numpy as np
import cv2
import os
import glob


def crop():
    src_dir = '/data/datasets/tianji/heyue/合约截图(1)(1)'
    dst_dir = '/data/datasets/tianji/heyue/crop_heyue'
    for img_path in glob.glob(os.path.join(src_dir, '*', '*')):
        img = cv2.imread(img_path)
        h, w, c = img.shape
        out_img = img[int(2 / 3 * h):, :int(w / 2), :]
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_path)), out_img)
# crop()

def get_crop(img, r_boxes, dst_dir):
    # img_path = '/data/datasets/tianji/heyue/crop_heyue/13506756093.jpg'
    # dst_dir = '/data/tmp'

    # img = cv2.imread(img_path)
    y_list = []
    for i, box in enumerate(r_boxes):
        y_list.append(box[:, 1].max())

    temp=[]

    for i in range(3):
        temp.append(y_list.index(max(y_list)))

        y_list[y_list.index(max(y_list))]=0

    print(temp)

    for ind in temp:
        box = r_boxes[ind]
        ul = (box[:, 0].min(), box[:, 1].min())
        br = (box[:, 0].max(), box[:, 1].max())
        out_img = img[ul[1]:br[1], ul[0]:br[0], :]
        cv2.imwrite(os.path.join(dst_dir, '{}.jpg'.format(ind)), out_img)
