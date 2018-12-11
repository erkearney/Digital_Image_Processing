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

This article helped me with pedestrian detection: https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

As mentioned, Tovio Roberts provided tremendous help to me for this project.
Here is his GitHub: https://github.com/clownfragment """

import cv2                              # For image and video analysis
import numpy as np                      # For matrix/vector manipulation
import pandas as pd                     # For reading and analyzing the GPS data
from matplotlib import pyplot as plt    # For plotting and visualizing
import argparse                         # For creating and handling command line arguments
from pathlib import Path                # For reading file paths
import imutils                          # For predestrial detection

parser = argparse.ArgumentParser(
    description="Vehicle crash and pedestrian detector built using OpenCV, numpy, and pandas"
)
parser.add_argument("-v", "--video", help="path to the input video")
parser.add_argument("-d", "--data", help="path to the GPS data")
parser.add_argument("-p", "--plot", action="store_true", help="Show the plot of the x, y, and z acceleration")
parser.add_argument("-s", "--show", action="store_true", help="Show the part of the video right before the crash")
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
        else:
            processed_data.loc[[row-startrow], "frame"] = frame
            processed_data.loc[[row-startrow], "timestamp"] = timestamp
            # Process the S data
            split = str(data.iloc[row]).split("\\t")
            z_accel = split[3].split("\n")[0]
            processed_data.loc[[row-startrow], "x_accel"] = float(split[1])
            processed_data.loc[[row-startrow], "y_accel"] = float(split[2])
            processed_data.loc[[row-startrow], "z_accel"] = float(z_accel)
            frame += 3

    processed_data = processed_data.dropna(how="all") # Drop empty rows
    return processed_data

def plot_data(data, title):
    """ Plots data, which should be a pandas dataframe returned by one of
    the other methods in this program. """
    data.plot.line()
    plt.title(title)
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

def open_video_at_frame(input_video, frame_num):
    # Initalize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("Opening the video to right before the crash, press 'q' to exit")
    cap = cv2.VideoCapture(str(input_video))
    if (cap.isOpened() == False):
        print("ERROR, could not open video: {}".format(input_video))

    cap.set(1, frame_num)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            i += 1
            if i > 1400 and i < 1415:
                cv2.imwrite('frame_' + str(i) + '.jpg', frame)
                # Detect people in the frame
                (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4),
                    padding=(8,8), scale=1.05)

                # Draw the bounding boxes
                for (x, y, w, h) in rects:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imwrite('frame_' + str(i) + 'bounded.jpg', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()

    cv2.destroyAllWindows()

def main():
    input_video, input_data = get_inputs()
    processed_data = process_GPS_data(input_data)
    delta = compute_delta(processed_data)
    # The time with the greatest delta value is most likely to be the time of the crash
    crash_index = delta["delta_y"].idxmax() 
    print("I think the crash happened here:")
    print(processed_data.iloc[crash_index])
    frame_num = processed_data.iloc[crash_index].frame
    if args.plot:
        # The frames were being plotted with the x, y, and z acceleration, 
        # so I drop that column before plotting, not ideal obviously, but
        # works for now ...
        processed_data = processed_data.drop(columns=["frame"], axis=1)
        plot_data(processed_data, "x, y, and z acceleration")
        plot_data(delta, "delta x, y, and z, and the sum of them all")
    if args.show:
        open_video_at_frame(input_video, frame_num)


# Globals
startrow = 20 # Offset to compensate for the fact we threw out some early data
G_row = startrow # G_row is each row that contains G_data
rows_btwn_G_rows = 11 # Number of rows between each G_row
start_frame = 6 # Offset to compensate for the fact we threw out some early data

if __name__ == "__main__":
    main()
