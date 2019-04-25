import argparse
import glob
import json
import os
import shlex
import subprocess
import sys

def generate_video_metadata(basedir, suffix):
    
    """Scans through directory, looking for video files that match
    the extension given by 'suffix'.  Uses ffprobe to look into the video
    files to obtain video format metadata such as height, width, fps,
    etc.

    Return an array of metadata, one element for each video found in the
    source directory. 

    Although video filenames are unique identifiers, this script also
    assigns each video a unique integer id.

    At this time there is not support for recurive directory tree
    search.  This would be a useful feature to add in the future.

    """

    vids = []
    
    for vid_filename in glob.glob(os.path.join(basedir, "*.%s" % suffix)):

        cmd = "ffprobe -v quiet -print_format json -show_streams %s" % vid_filename;
        outp = subprocess.check_output(shlex.split(cmd)).decode("utf-8")
        streams = json.loads(outp)["streams"]
        video_stream = [s for s in streams if s["codec_type"] == "video"][0]

        [num, denom] = map(int, video_stream["r_frame_rate"].split('/'))
        fps = float(num) / float(denom)
        num_frames = video_stream["nb_frames"]
        width = video_stream["width"]
        height = video_stream["height"]

        id = len(vids)
        fname = os.path.basename(vid_filename)
        
        meta = { "id" : id,
                 "filename" : fname,
                 "num_frames" : num_frames,
                 "fps" : fps,
                 "width" : width,
                 "height" : height}

        vids.append(meta)

    return vids



parser = argparse.ArgumentParser(description='Generate video metadata.')
parser.add_argument('-s', '--suffix', default='mp4', help='Suffix for video files (default=mp4)')
parser.add_argument('-r', '--recursive', default=False, action='store_true', help='Perform recursive search for videos')
parser.add_argument('basedir', help='Base directory containing videos')
parser.add_argument('outfile', help='Output json file containing metadata for all videos')

args = parser.parse_args()
basedir = args.basedir
outfile = args.outfile
suffix = args.suffix
recursive = args.recursive

if recursive:
    print("ERROR: recursive option not supported at the moment.")
    sys.exit(-1);

# dump json output
with open(outfile, 'w') as file:
    print("Generating metadata from video files in: %s  (ext=.%s)" % (basedir, suffix))
    meta = generate_video_metadata(basedir, suffix)
    print("Found %d videos." % len(meta))
    print("Writing %s" % outfile)
    json.dump(meta, file)

    
