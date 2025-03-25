"""
Generates a ChArUco board for extrinsic calibration
"""

import cv2 as cv
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ARUCO_DICT = cv.aruco.DICT_6X6_250
ARUCO_DICT = cv.aruco.DICT_4X4_50

SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 3
SQUARE_LENGTH = 0.15  # arbitrary unit
MARKER_LENGTH = 0.13  # same unit

LENGTH_PX = 1920  # total length of the page in pixels, wont matter much as it will be a scalable pdf
MARGIN_PX = 20  # size of the margin in pixels
SAVE_NAME = "ChArUco_Marker.png"


dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv.aruco.CharucoBoard(
    (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary
)

size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
img = cv.aruco.CharucoBoard.generateImage(
    board, (LENGTH_PX, int(LENGTH_PX * size_ratio)), marginSize=MARGIN_PX
)
cv.imwrite(SAVE_NAME, img)
im = Image.fromarray(np.uint8(img))
im.save(
    f"./pattern_{SQUARES_HORIZONTALLY}x{SQUARES_VERTICALLY}.pdf",
    "PDF",
    resolution=100.0,
)
