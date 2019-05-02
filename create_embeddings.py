from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import glob
import time
import random
import argparse

from tqdm import tqdm, tqdm_notebook
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from ConfigParser import SafeConfigParser
config = SafeConfigParser()
config.read('tensorflow_config.ini')
slim_dir = config.get('main', 'slim_dir')
pretrain_ckpt = config.get('main', 'pretrain_ckpt')
sys.path.append(os.path.realpath(slim_dir))

from nets import resnet_v2
from preprocessing import inception_preprocessing
from datasets import imagenet

## Parameters from the Mask R-CNN detectron
id2name = {0: u'person', 1: u'bicycle', 2: u'car', 3: u'motorcycle', 4: u'airplane',
          5: u'bus', 6: u'train', 7: u'truck', 8: u'boat', 9: u'traffic light',
          10: u'fire hydrant', 11: u'stop sign', 12: u'parking meter', 13: u'bench',
          14: u'bird', 15: u'cat', 16: u'dog', 17: u'horse', 18: u'sheep', 19: u'cow',
          20: u'elephant', 21: u'bear', 22: u'zebra', 23: u'giraffe', 24: u'backpack',
          25: u'umbrella', 26: u'handbag', 27: u'tie', 28: u'suitcase', 29: u'frisbee',
          30: u'skis', 31: u'snowboard', 32: u'sports ball', 33: u'kite', 34: u'baseball bat',
          35: u'baseball glove', 36: u'skateboard', 37: u'surfboard', 38: u'tennis racket',
          39: u'bottle', 40: u'wine glass', 41: u'cup', 42: u'fork', 43: u'knife',
          44: u'spoon', 45: u'bowl', 46: u'banana', 47: u'apple', 48: u'sandwich',
          49: u'orange', 50: u'broccoli', 51: u'carrot', 52: u'hot dog', 53: u'pizza',
          54: u'donut', 55: u'cake', 56: u'chair', 57: u'couch', 58: u'potted plant',
          59: u'bed', 60: u'dining table', 61: u'toilet', 62: u'tv', 63: u'laptop',
          64: u'mouse', 65: u'remote', 66: u'keyboard', 67: u'cell phone',
          68: u'microwave', 69: u'oven', 70: u'toaster', 71: u'sink', 72: u'refrigerator',
          73: u'book', 74: u'clock', 75: u'vase', 76: u'scissors', 77: u'teddy bear',
          78: u'hair drier', 79: u'toothbrush', 80: u'banner', 81: u'blanket',
          82: u'bridge', 83: u'cardboard', 84: u'counter', 85: u'curtain',
          86: u'door-stuff', 87: u'floor-wood', 88: u'flower', 89: u'fruit',
          90: u'gravel', 91: u'house', 92: u'light', 93: u'mirror-stuff', 94: u'net',
          95: u'pillow', 96: u'platform', 97: u'playingfield', 98: u'railroad',
          99: u'river', 100: u'road', 101: u'roof', 102: u'sand', 103: u'sea',
          104: u'shelf', 105: u'snow', 106: u'stairs', 107: u'tent', 108: u'towel',
          109: u'wall-brick', 110: u'wall-stone', 111: u'wall-tile', 112: u'wall-wood',
          113: u'water-other', 114: u'window-blind', 115: u'window-other',
          116: u'tree-merged', 117: u'fence-merged', 118: u'ceiling-merged',
          119: u'sky-other-merged', 120: u'cabinet-merged', 121: u'table-merged',
          122: u'floor-other-merged', 123: u'pavement-merged', 124: u'mountain-merged',
          125: u'grass-merged', 126: u'dirt-merged', 127: u'paper-merged', 128: u'food-other-merged',
          129: u'building-other-merged', 130: u'rock-merged', 131: u'wall-other-merged', 132: u'rug-merged'}

bbox_idx, class_idx, score_idx = 0, 1, 2
y1_idx, x1_idx, y2_idx, x2_idx = 0, 1, 2, 3

## Arguments parser
parser = argparse.ArgumentParser(description=\
  'Generate JSON (containing bbox info + id) and embedding binary from video and Mask R-CNN dump.')
parser.add_argument('video', help='Input video file')
parser.add_argument('maskrcnn_dump', help='Input Mask R-CNN detectron file (.npy)')
parser.add_argument('-e', '--embedding', default=None, \
  help='Output binary file containing embeddings (default videoname_param_embeddings.npy)')
parser.add_argument('-j', '--json', default=None, \
  help='Output JSON file containing bounding box information (default videoname_param_bboxes.json)')
parser.add_argument('-x', '--minx', default=8, type=int, \
  help='Minimum width of the bounding boxes to be considered (default 8)')
parser.add_argument('-y', '--miny', default=8, type=int, help=\
  'Minimum height of the bounding boxes to be considered (default 8)')
parser.add_argument('-s', '--score', default=0.0, type=float, \
  help='Minimum score of the bounding boxes to be considered (default 0.000)')
parser.add_argument('-g', '--gpuid', default=0, type=int, \
  help='Claim this gpu id for tensorflow (default 0)')

args = parser.parse_args()

## Set appropriate paths
param_str = "_" + str(args.minx) + "_" + str(args.miny) + "_" + str(args.score) + "_"
if args.embedding is None:
    embedding_path = args.video[:-4] + param_str + "embeddings.npy"
else:
    embedding_path = args.embedding
if args.json is None:
    json_path = args.video[:-4] + param_str +  "_bboxes.json"
else:
    json_path = args.json

## Compute the json and embeddings
with tf.Graph().as_default():
    # set up the resnet pretrained model
    image_size = 299
    names = imagenet.create_readable_names_for_imagenet_labels()
    image = tf.placeholder(tf.uint8, (None, None, 3))
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_image = tf.expand_dims(processed_image, 0)
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_101(processed_image, 1001, is_training=False)
        pool5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_101/pool5:0")
    
    init_fn = slim.assign_from_checkpoint_fn(pretrain_ckpt,
                            slim.get_model_variables('resnet_v2'))

    # set gpu id
    gpu_options = tf.GPUOptions(visible_device_list=str(args.gpuid))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init_fn(sess)

        # load the bbox data
        data = np.load(args.maskrcnn_dump, allow_pickle=True)[()]
        # load the video and set width and height
        vid_obj = cv2.VideoCapture(args.video) 
        vid_width = vid_obj.get(3)
        vid_height = vid_obj.get(4)

        json_obj = []
        embeddings_list = []
        bbox_unique_id = 0

        # for all the frames, each element of data contains info of 1 frame
        for idx in tqdm(range(len(data))):
            success, img = vid_obj.read() 

            curr_dict = {"video": args.video, "frame": idx, "bboxes": []}

            # for all the bounding boxes
            for jdx in range(len(data[idx][bbox_idx])):

                # considering only boxes that pass the given score threshold
                if float(data[idx][score_idx][jdx]) > args.score:
                    curr_bbox = {"x1": float(data[idx][bbox_idx][jdx][x1_idx]),
                                    "y1": float(data[idx][bbox_idx][jdx][y1_idx]),
                                    "x2": float(data[idx][bbox_idx][jdx][x2_idx]),
                                    "y2": float(data[idx][bbox_idx][jdx][y2_idx]),
                                    "class": str(id2name[data[idx][class_idx][jdx] - 1]),
                                    "score": float(data[idx][score_idx][jdx])
                                    }

                    # scale the coordinates since the mask-rcnn dumpled is normalized
                    x1 = int(curr_bbox['x1'] * vid_width)
                    x2 = int(curr_bbox['x2'] * vid_width)
                    y1 = int(curr_bbox['y1'] * vid_height)
                    y2 = int(curr_bbox['y2'] * vid_height)

                    # consider only the boxes that pass the size constraints given
                    if y2 - y1 >= args.miny and x2 - x1 >= args.minx:
                        patch = img[y1:y2, x1:x2, :]
                        scaled_img, logit_vals, embedding = sess.run([processed_image, logits, pool5], \
                                                                    feed_dict={image: patch})
                        curr_bbox['bbox_id'] = bbox_unique_id
                        bbox_unique_id += 1
                        embeddings_list.append(embedding[0, 0, 0, :])
                        curr_dict["bboxes"].append(curr_bbox)

            json_obj.append(curr_dict)

        # dump the json
        json_file = open(json_path, "w")
        json_file.write(json.dumps(json_obj))
        json_file.close()
 
        # dump the embeddings
        np.save(embedding_path, embeddings_list)
