import os
import math
import glob
import json
import datetime
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# reprojection error measured in pixel on the image plane


# ChArUco board measurements #
# square length:     227 mm
# marker length:     197 mm
# square-marker gap: 10 mm


def get_SN(image_name: str) -> str:
    """
    Get the camera serial number from image name.
    Expects serial number to be the first part of the name, separated by '_'
    """
    return image_name.split("_")[1].split("\\")[1]


# charuco board parameters
ARUCO_DICT = cv.aruco.DICT_4X4_50
SQUARES_VERTICALLY = 5  # num squares vertical direction
SQUARES_HORIZONTALLY = 3  # num squares horizontal direction
SQUARE_LENGTH = 227  # in arbitrary unit (base for all calculations)
MARKER_LENGTH = 197  # same unit

# outer corners of the ChArUco pattern (for visualization only)
board_corners = [
    [0, 0, 0],
    [0, SQUARE_LENGTH * SQUARES_HORIZONTALLY, 0],
    [SQUARE_LENGTH * SQUARES_VERTICALLY, SQUARE_LENGTH * SQUARES_HORIZONTALLY, 0],
    [SQUARE_LENGTH * SQUARES_VERTICALLY, 0, 0],
]

# define charuco board
aruco_dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv.aruco.CharucoBoard(
    (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dictionary,
)

# load factory-calibration data for all cameras
with open("./camera_params.json", "r") as f:
    intrinsic_params = json.load(f)


input_dir = "./charuco_data/*.jpg"
output_dir = "./charuco_output/"
json_save_dir = "./"

all_camera_coords = []
calibration_results = {}


for file in glob.glob(input_dir):

    # read in image, get the serial number
    gray_img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    marker_img = cv.imread(file)

    cam_serial_number = get_SN(file)

    calibration_results[cam_serial_number] = {}

    # select the corresponding factory-calibration data
    camera_matrix = np.array(
        intrinsic_params[cam_serial_number]["left_sensor"]["camera_matrix"]
    )
    dist_coeffs = np.array(
        intrinsic_params[cam_serial_number]["left_sensor"]["distortion_coeff"]
    )

    imgSize = gray_img.shape

    # Undistort the image
    undistorted_image = cv.undistort(gray_img, camera_matrix, dist_coeffs)
    marker_img = cv.undistort(marker_img, camera_matrix, dist_coeffs)
    corners_img = marker_img.copy()

    cv.imwrite(
        output_dir + file.split(".")[1].split("\\")[1] + "_undistorted.jpg",
        undistorted_image,
    )

    # Detect markers in the undistorted image
    marker_corners, marker_ids, _ = cv.aruco.detectMarkers(
        undistorted_image, aruco_dictionary, parameters=cv.aruco.DetectorParameters()
    )
    # draw the detected markers
    cv.aruco.drawDetectedMarkers(marker_img, marker_corners, marker_ids)

    if marker_ids is not None:
        # detect the corners of the chessboard pattern (corner = junciton of 2 dark squares)
        enough_corners, chessboard_corners, chessboard_corner_ids = (
            cv.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, undistorted_image, board
            )
        )
        # obj_points = np.array([[]])
        # img_points = np.array([[]])
        # board.matchImagePoints(marker_corners, marker_ids, obj_points, img_points)

        # if corners are found, refine them
        if chessboard_corners is not None:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 50, 0.1)
            cv.cornerSubPix(
                undistorted_image, chessboard_corners, (5, 5), (-1, -1), term
            )

        # draw the detected corners
        cv.aruco.drawDetectedCornersCharuco(
            corners_img, chessboard_corners, chessboard_corner_ids
        )

        # If enough corners are found, estimate the pose
        if enough_corners:
            print(f"\n \n \n {file}")
            # calculate extrinsics
            success, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(
                chessboard_corners,
                chessboard_corner_ids,
                board,
                camera_matrix,
                dist_coeffs,
                None,
                None,
            )

            # If pose estimation is successful, draw the axis at the origin
            if success:
                cv.drawFrameAxes(
                    corners_img,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    length=100,
                    thickness=5,
                )

            # calculating rotation matrix, camera position
            R, jac = cv.Rodrigues(rvec)
            camera_position = -np.matrix(R).T * np.matrix(tvec)
            all_camera_coords.append((cam_serial_number, camera_position))

            # distance of camera from origin
            distance_mm = round(np.linalg.norm(camera_position), 3)

            calibration_results[cam_serial_number]["time"] = str(
                datetime.datetime.now()
            )
            calibration_results[cam_serial_number]["input_filename"] = file
            calibration_results[cam_serial_number][
                "camera_position"
            ] = camera_position.tolist()

            calibration_results[cam_serial_number]["distance_mm"] = distance_mm
            calibration_results[cam_serial_number]["rvec"] = rvec.tolist()
            calibration_results[cam_serial_number]["tvec"] = tvec.tolist()

            print(
                f"distance: {math.floor(distance_mm/1000)} m, {round((distance_mm-math.floor(distance_mm/1000)*1000)/10,2)} cm "
            )
            print(f"camera coords: \n{camera_position}")

    else:
        print(f"cant find markers for image: {file}")

    # save image of detected markers
    cv.imwrite(
        output_dir + file.split(".")[1].split("\\")[1] + "markers.jpg",
        marker_img,
    )
    # save image of detected corners
    cv.imwrite(
        output_dir + file.split(".")[1].split("\\")[1] + "corners.jpg", corners_img
    )

    # flags that fix every intrinsic parameter, we only want the reprojection error using the factory settings
    calib_flags = (
        cv.CALIB_USE_INTRINSIC_GUESS
        + cv.CALIB_FIX_ASPECT_RATIO
        + cv.CALIB_FIX_PRINCIPAL_POINT
        + cv.CALIB_FIX_FOCAL_LENGTH
        + cv.CALIB_FIX_K1
        + cv.CALIB_FIX_K2
        + cv.CALIB_FIX_K3
        + cv.CALIB_FIX_K4
        + cv.CALIB_FIX_K5
        + cv.CALIB_FIX_K6
    )

    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv.aruco.calibrateCameraCharuco(
            [chessboard_corners],
            [chessboard_corner_ids],
            board,
            undistorted_image.shape[:2],
            calib_flags,
            None,
        )
    )
    calibration_results[cam_serial_number]["reprojection_error"] = reprojection_error

    print(f"\nReprojection error: {round(reprojection_error,3)} pixels")

# save calibration results
with open(os.path.join(json_save_dir, "calibration_results.json"), "w") as outfile:
    outfile.write(json.dumps(calibration_results, indent=4))


## Drawing everything
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_aspect("auto")
# ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim(-6000, 6000)
ax.set_ylim(-6000, 6000)
ax.set_zlim(-4000, 0)


ax.plot3D(0, 0, 0, "mo", markersize=3, label="origin")

# drawing the board
for idx in range(-1, len(board_corners) - 1):
    label = "_nolegend_"
    if idx == 0:
        label = "calib pattern"
    ax.plot(
        [board_corners[idx][0], board_corners[idx + 1][0]],
        [board_corners[idx][1], board_corners[idx + 1][1]],
        [0, 0],
        "g-",
        label=label,
    )

# drawing cameras
cam_styles = ["rv", "rs", "ro", "rd"]
for idx, (cam_name, coords) in enumerate(all_camera_coords):
    label = cam_name
    ax.plot3D(coords[0], coords[1], coords[2], cam_styles[idx], label=label)

ax.view_init(elev=-145, azim=60)
ax.legend()
plt.show()
