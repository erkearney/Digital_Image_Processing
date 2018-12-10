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

import cv2                              # For image and video analysis
import numpy as np                      # For matrix/vector manipulation
import pandas as pd                     # For reading and analyzing the GPS data
from matplotlib import pyplot as plt    # For plotting and visualizing
import argparse                         # For creating and handling command line arguments
from pathlib import Path                # For reading file paths

parser = argparse.ArgumentParser(
    description="Vehicle crash and pedestrian detector built using OpenCV, numpy, and pandas"
)
parser.add_argument("-v", "--video", help="path to the input video")
parser.add_argument("-d", "--data", help="path to the GPS data")
parser.add_argument("-p", "--plot", action="store_true", help="Show the plot of the x, y, and z acceleration")
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
    """ Converts the GPS data into a more useful format, consisting of:
    frame | timestamp | x-accel | y_accel | z_accel """
    global startrow
    global G_row
    global rows_btwn_G_rows
    global start_frame
    frame = start_frame
    data = pd.read_csv(input_data, header=None)

    col_names = ["frame", "timestamp", "x_accel", "y_accel", "z_accel"]
    processed_data = pd.DataFrame(index=range(len(data)-startrow) , columns=col_names)
    
    for row in range(startrow, len(data)):
        if row == G_row:
            # Process the G data
            split = str(data.iloc[row]).split("\\t")
            timestamp = (split[1].split(" ")[1])
            G_row += rows_btwn_G_rows
            frame += 1
        else:
            processed_data.loc[[row-startrow], "frame"] = frame
            processed_data.loc[[row-startrow], "timestamp"] = timestamp
            # Process the S data
            split = str(data.iloc[row]).split("\\t")
            z_accel = split[3].split("\n")[0]
            processed_data.loc[[row-startrow], "x_accel"] = float(split[1])
            processed_data.loc[[row-startrow], "y_accel"] = float(split[2])
            processed_data.loc[[row-startrow], "z_accel"] = float(z_accel)

    processed_data = processed_data.dropna(how="all") # Drop empty rows
    return processed_data

def plot_data(processed_data):
    """ Plots the x-accel, y_accel, and z_accel over time
    processed_data should be a pandas dataframe returned by 
    process_GPS_data() """
    processed_data.plot.line()
    plt.title("x, y, and z acceleration over time")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.show()

def compute_delta(processed_data):
    col_names = ["delta_sum", "delta_x", "delta_y", "delta_z"]
    delta = pd.DataFrame(index=range(1), columns = col_names)
    previous_x = previous_y = previous_z = None
    for index, row in processed_data.iterrows():
        x, y, z = float(row["x_accel"]), float(row["y_accel"]), float(row["z_accel"])
        if previous_x == None:
            previous_x, previous_y, previous_z = x, y, z
        else:
            delta_x, delta_y, delta_z = abs(x - previous_x), abs(y - previous_y), abs(z - previous_z)
            delta_sum = delta_x + delta_y + delta_z
            new_row = pd.DataFrame([[delta_sum, delta_x, delta_y, delta_z]], columns = col_names)
            delta = delta.append(new_row, ignore_index=True)

    delta = delta.dropna(how="all")
    return delta

def main():
    input_video, input_data = get_inputs()
    processed_data = process_GPS_data(input_data)
    if args.plot:
        plot_data(processed_data)
    delta = compute_delta(processed_data)
    print(delta["delta_sum"].idxmax())

# Globals
startrow = 20 # Offset to compensate for the fact we threw out some early data
G_row = startrow # G_row is each row that contains G_data
rows_btwn_G_rows = 11 # Number of rows between each G_row
start_frame = 6 # Offset to compensate for the fact we threw out some early data

if __name__ == "__main__":
    main()
