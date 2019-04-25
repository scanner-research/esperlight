# Esperlight

This repo contains simple scripts to get up and running on a new video dataset.  The idea here is to depend only on our basic Esper-ecosystem tools: vgrid, rekall, etc.  and otherwise exchange data between these tools in simple formats like json.

# Generating video metadata

The first step in getting a video dataset ready for manipulation is to produce a list of videos in the collection.  This list will contain metadata about videos that is needed for visualization purposes.  Esperlight contains a script `create_video_metadata.py` that generates this list from a directory of videos.  For example, if you have a directory of `.mp4` videos named `VIDEO_SOURCE_DIR`, the following command line generates a json file that contains basic information about all the videos (height, width, fps, duration, ect.)

    python create_video_metadata.py VIDEO_SOURCE_DIR myvideolist.json --suffix mp4

_At the moment, `create_video_metadata.py` does not support recursive search of the directory tree.  This would be a reasonable addition in the future since I expect that many datasets we download from the internet will have subdirectory structure._

As an example, running `create_video_metadata.py` on the directory of videos containing the [__kayvon10__](https://olimar.stanford.edu/hdd/kayvon10/) dataset, yields the video metadata file you see here: <https://olimar.stanford.edu/hdd/kayvon10/kayvon10.json>

# Visualizing the videos

__TODO: ask Dan to point to his notebook here.__ 
