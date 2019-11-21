# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import sys
from pathlib import Path
from pdf2image import convert_from_path

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
            elif ext == '.pdf':
                pages = convert_from_path(os.path.join(dirpath, file))
                for i, page in enumerate(pages):
                    p_name = filename + "_" + str(i) + ".jpg"
                    page.save(os.path.join(dirpath, p_name), 'JPEG')
                    img_files.append(os.path.join(dirpath, p_name))
                    
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files