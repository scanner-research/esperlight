import os
import sys
import json
import numpy as np
from tqdm import tqdm

bbox_idx, class_idx, score_idx = 0, 1, 2
x1_idx, y1_idx, x2_idx, y2_idx = 1, 0, 3, 2

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

video_dir = str(sys.argv[1])
video_list = []
for file in os.listdir(video_dir):
    if file.endswith(".mp4"):
        video_list.append(file)

for video_name in tqdm(video_list):
    data = np.load(os.path.join(video_dir, 'detectron_large_mask_rcnn_1_' + video_name[:-4] + '.npy'), allow_pickle=True)[()]
    # data = [data[0]]

    json_obj = []
    for idx in range(len(data)):
        curr_dict = {}
        curr_dict["video"] = video_name
        curr_dict["frame"] = idx
        curr_dict["bboxes"] = []
        for jdx in range(len(data[idx][bbox_idx])):
            curr_bbox = {"x1": (data[idx][bbox_idx][jdx][x1_idx]),
                            "y1": (data[idx][bbox_idx][jdx][y1_idx]),
                            "x2": (data[idx][bbox_idx][jdx][x2_idx]),
                            "y2": (data[idx][bbox_idx][jdx][y2_idx]),
                            "class": str(id2name[data[idx][class_idx][jdx] - 1]),
                            "score": (data[idx][score_idx][jdx])
                            }
            curr_dict["bboxes"].append(curr_bbox)
        json_obj.append(curr_dict)
        # break

    json_file = open(os.path.join(video_dir, video_name[:-4] + ".json"), "w")
    json_file.write(json.dumps(json_obj))
    json_file.close()
