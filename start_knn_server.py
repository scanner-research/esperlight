from flask import Flask
import os
import json

import faiss
import numpy as np

from flask import jsonify
from flask import request

app = Flask(__name__) 
xb, video_dict, index = None, None, None

@app.route('/setup', methods = ['POST'])
def setup_knn():
	global xb, video_dict, index
	try:
	    jsondata = request.get_json()
	    data = json.loads(jsondata)
	    video_dir = data['video_dir']
	    minx = data['minx']
	    miny = data['miny']
	    score = data['score']
	    
	    video_list = []
	    for file in os.listdir(video_dir):
	        if file.endswith(".mp4"):
	            video_list.append(file)
	    
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
	    
	    d = len(xb[0])
	    index = faiss.IndexFlatL2(d)
	    index.add(xb)

	except:
		result = {'error': True}

	return jsonify(result)

def convert_idx_to_vidinfo(idx, video_dict):
    curr_sum = 0
    for x in video_dict:
        if idx < curr_sum + x[1]:
            return (x[0], idx - curr_sum)
        else:
            curr_sum = x[1] + curr_sum

def convert_vidinfo_to_idx(video_name, bbox_id):
    curr_sum = 0
    for x in video_dict:
        if x[0] == video_name:
            return curr_sum + bbox_id
        else:
            curr_sum = curr_sum + x[1]


@app.route('/knn', methods = ['POST'])
def find_knn():
	global xb, video_dict, index
	try:
	    jsondata = request.get_json()
	    data = json.loads(jsondata)
	    video_name = data['video_name']
	    bbox_id = data['bbox_id']
	    k = data['k']
	    
	    xq = np.asarray([xb[convert_vidinfo_to_idx(video_name, bbox_id)]])
	    D, I = index.search(xq, k)     # actual search
	    print(I)
	    bboxes = []
	    for idx in I[0]:
	        bboxes.append(convert_idx(idx, video_dict))
	    reesult = {'success': True. 'bboxes': bboxes} 
	except:
		result = {'error': True}

	return jsonify(result)

if __name__ == '__main__':
    app.run()