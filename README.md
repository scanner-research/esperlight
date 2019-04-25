# Esperlight

This repo contains simple scripts to get up and running on a new video dataset.  The idea here is to provide a minimal set of functionality that depends only on our basic Esper-ecosystem tools: vgrid, rekall, etc. and to exchange data between these tools in simple formats like json.

# Generating video metadata

The first step in getting a video dataset ready for manipulation is to produce a list of videos in the collection.  This list will contain metadata about videos that is needed for visualization purposes.  The Esperlight script `create_video_metadata.py` generates this list from a directory of videos.  For example, if you have a directory of `.mp4` videos named `VIDEO_SOURCE_DIR`, the following command line generates a json file that contains basic information about all the videos (height, width, fps, duration, etc.)

    python create_video_metadata.py VIDEO_SOURCE_DIR myvideolist.json --suffix mp4

_At the moment, `create_video_metadata.py` does not support recursive search of the source directory tree.  This would be a reasonable addition in the future since I expect that many datasets we download from the internet will have subdirectory structure._

As an example, running [`create_video_metadata.py`](view_collection.ipynb) on the directory of videos containing the [__kayvon10__](https://olimar.stanford.edu/hdd/kayvon10/) dataset, yields the video metadata file you see here: <https://olimar.stanford.edu/hdd/kayvon10/kayvon10.json>

# Visualizing a collection of videos

Now let's take a look at all the videos. Take a look at the notebook [`view_collection.ipynb`](view_collection.ipynb) for a demonstration of how to read the video metafile file, and use the [Vgrid](https://github.com/scanner-research/vgrid) widget to display the videos.  

Note that running this notebook depends on rekall, vgrid, and vgrid_jupyter.
