""" CS390S -- Digital Image Processing -- Professor Feng Jiang
Project 3 -- Realistic Driving Analysis
Written in Python by Eric Kearney on December 9th, 2018, with great help from
Tovio Roberts. 

The purpose of this project is to be able to automatically detect a car
accident using video footage and GPS sensor data, it should also be able to
detect pedestrians in the video footage.

Due to its large filesize, the video, "0.MOV", will not be included on my
GitHub. It can be found here:
https://metrostate-bb.blackboard.com/webapps/blackboard/content/listContent.jsp?course_id=_131037_1&content_id=_7094184_1

As mentioned, Tovio Roberts provided tremendous help to me for this project.
Here is his GitHub: https://github.com/clownfragment """

import cv2                  # For image and video analysis
import numpy as np          # For matrix/vector manipulation
import pandas as pd         # For reading and analyzing the GPS data
import argparse             # For creating and handling command line arguments
from pathlib import Path    # For reading file paths

parser = argparse.ArgumentParser(
    description="Vehicle crash and pedestrian detector built using OpenCV, numpy, and pandas"
)
parser.add_argument("-v", "--video", help="path to the input video")
parser.add_argument("-d", "--data", help="path to the GPS data")
args = parser.parse_args()

def setup():
    if args.video:
        input_video = Path(args.video)
        print("Using {} as the input video".format(input_video))
        if not input_video.is_file():
            print("ERROR: {} not found".format(input_video))
            exit(1)
    else:
        input_video = Path("./0.MOV")
        print("Using the default, {} as the input video".format(input_video))
        if not input_video.is_file():
            print("ERROR: {} not found".format(input_video))
            print("You may need to download the default video file, make sure you put it in the same directory as this program:")
            print("https://metrostate-bb.blackboard.com/webapps/blackboard/content/listContent.jsp?course_id=_131037_1&content_id=_7094184_1")
            exit(1)

    input_data = Path("./000.dat")
    if args.data:
        input_data = Path(args.data)
    print("Using {} as the GPS data".format(input_data))
    if not input_data.is_file():
        print("ERROR: {} not found".format(input_data))
        exit(1)

def main():
    setup()

if __name__ == "__main__":
    main()
