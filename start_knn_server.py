import os
import json
import argparse

import faiss
import numpy as np

from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__) 

## Arguments parser
from ConfigParser import SafeConfigParser
config = SafeConfigParser()
config.read('knn_server_config.ini')
video_dir = config.get('main', 'video_dir')
minx = config.get('main', 'minx')
miny = config.get('main', 'miny')
score = config.get('main', 'score')

## Store the params to be returned for `status`
params = {"minx": minx, "miny": miny, "score": score}

## Find the videos 
video_list = []
for file in os.listdir(video_dir):
    if file.endswith(".mp4"):
        video_list.append(file)

## Prepare data for knn-search
video_dict = []
first = True
for video in video_list[:3]:
    curr_data = np.load(os.path.join(video_dir, video[:-4] + '_' + str(minx) + '_' + 
        str(miny) + '_' + str(score) + '_' + 'embeddings.npy'), allow_pickle=True)
    video_dict.append((video, len(curr_data)))
    if first:
        xb = curr_data
        first = False
    else:
        xb = np.concatenate((xb, curr_data), axis=0)

## Create faiss index
d = len(xb[0])
index = faiss.IndexFlatL2(d)
index.add(xb)

## Check the status of the server by sending a POST request to /status
## Returns list of videos along with the parameters if success
## Else returns error 
@app.route('/status', methods = ['POST'])
def setup_knn():
    global video_dict, params
    try:
        videos = []
        for x in video_dict:
            videos.append(x[0])
        result = {'success': True, 'videos': videos, 'minx': params["minx"], \
                'miny': params["miny"], 'score': params["score"]}
    except:
        result = {'error': True}

    return jsonify(result)

## Knn search here works by creating a mega array got by concatenating bbox 
## embeddings from all videos. Every request bbox is converted to an index
## to the mega array, knn-search is run on this mega array and the returned
## indices are converted back to video_name & bbox_id.

## Convert video_name, bbox_id to an index in the concatenation of all bboxes
def convert_vidinfo_to_idx(video_name, bbox_id):
    curr_sum = 0
    for x in video_dict:
        if x[0] == video_name:
            return curr_sum + bbox_id
        else:
            curr_sum = curr_sum + x[1]

## Convert index in mega array back to video_name & bbox_id
def convert_idx_to_vidinfo(idx, video_dict):
    curr_sum = 0
    for x in video_dict:
        if idx < curr_sum + x[1]:
            return (x[0], idx - curr_sum)
        else:
            curr_sum = x[1] + curr_sum

## Run the knn search
@app.route('/knn', methods = ['POST'])
def find_knn():
    global xb, video_dict, index
    try:
        jsondata = request.get_json()
        data = json.loads(jsondata)
        video_name = data['video_name']
        bbox_id = data['bbox_id']
        k = data['k']

        query_idx = convert_vidinfo_to_idx(video_name, bbox_id)
        xq = np.asarray([xb[query_idx]])

        D, I = index.search(xq, k)
        print(I)
        bboxes = []
        for idx in I[0]:
            bboxes.append(convert_idx_to_vidinfo(idx, video_dict))
        result = {'success': True, 'bboxes': bboxes} 
    except:
        result = {'error': True}

    return jsonify(result)

if __name__ == '__main__':
    app.run()

