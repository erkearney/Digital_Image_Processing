import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Simple python script to resize images"
)
parser.add_argument("-i", "--image", nargs="+", required=True, help="path to the input image(s)")
parser.add_argument("-s", "--size", help="desired width (default=512)")
parser.add_argument("-S", "--square",action="store_true", help="square the image(s) (default=False)")
parser.add_argument("-e", "--ext", help="desired extension for resized image(s) (default=.jpg)")
args = parser.parse_args()

for i, image in enumerate(args.image):
    if not Path(image).is_file():
        print("Error, {} doesn't exist".format(image))
    else:
        img = cv2.imread(image)

    if args.size:
        S = int(args.size)
    else:
        S = 512

    height, width, depth = img.shape
    if args.square:
        img_w_scale = S/width
        img_h_scale = S/height
    else:
        img_w_scale, img_h_scale = S/width
    newX, newY = img.shape[1]*img_w_scale, img.shape[0]*img_h_scale
    img = cv2.resize(img, (int(newX), int(newY)))

    if args.ext:
        extension = args.ext
    else:
        extension = ".jpg"

    cv2.imwrite("./" + str(i) + extension, img)
