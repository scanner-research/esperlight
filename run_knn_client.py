import json
import requests

# check the status of the knn server
res = requests.post("http://charlotte.stanford.edu:8321/status").json()
print(res)

# request for knn 
payload = {"video_name": "074496ba-19237a5b.mp4", "bbox_id": 8, "k": 100}
s = json.dumps(payload)
print(s)
res = requests.post("http://charlotte.stanford.edu:8321/knn", json = s).json()
print(res)
