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
from matplotlib import pyplot as plt
import argparse             # For creating and handling command line arguments
from pathlib import Path    # For reading file paths

parser = argparse.ArgumentParser(
    description="Vehicle crash and pedestrian detector built using OpenCV, numpy, and pandas"
)
parser.add_argument("-v", "--video", help="path to the input video")
parser.add_argument("-d", "--data", help="path to the GPS data")
args = parser.parse_args()

def get_inputs():
    """ Get and verify user inputs, if there are no user inputs, verify and 
    use the default resources. """
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
    
    return input_video, input_data

def process_GPS_data(input_data):
    data = pd.read_csv(input_data, header=None)
    # Personally, the only data points I care about are the timestamp,
    # the frame number, and the x, y, and z acceleration, so let's
    # create a custom dataframe to meet these specifications
    startrow = 20 # Start at this row, some missing data in the rows before it

    col_names = ["frame", "timestamp", "x_accel", "y-accel", "z-accel"]
    processed_data = pd.DataFrame(index=range(len(data)-startrow) , columns=col_names)
    
    G_row = startrow # G_row is each row that contains G_data
    frame = 6 # Offset to compensate for the fact we threw out some early data
    for row in range(startrow, len(data)):
        if row == G_row:
            # Process the G data
            split = str(data.iloc[row]).split("\\t")
            timestamp = (split[1].split(" ")[1])
            G_row += 11
            frame += 1
        else:
            processed_data.loc[[row-startrow], "frame"] = frame
            processed_data.loc[[row-startrow], "timestamp"] = timestamp
            # Process the S data
            split = str(data.iloc[row]).split("\\t")
            z_accel = split[3].split("\n")[0]
            processed_data.loc[[row-startrow], "x_accel"] = float(split[1])
            processed_data.loc[[row-startrow], "y-accel"] = float(split[2])
            processed_data.loc[[row-startrow], "z-accel"] = float(z_accel)

    processed_data = processed_data.dropna(how="all") # Drop empty rows
    print(processed_data["x_accel"].describe())
    processed_data.plot.hist()
    plt.show()

def main():
    input_video, input_data = get_inputs()
    process_GPS_data(input_data)

if __name__ == "__main__":
    main()
