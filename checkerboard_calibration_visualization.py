import os
import json
import cv2 as cv
import numpy as np
from glob import glob
import os
import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class Pattern:
    """
    Chessboard pattern parameters
    """

    def __init__(self, height, width, square_size):
        self.pattern_height = height
        self.pattern_width = width
        self.square_size = float(square_size)
        self.pattern_type = "chessboard"
        self.pattern_size = (self.pattern_width, self.pattern_height)

        self.pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        self.pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size


def get_serial_number(filename: str) -> str:
    """
    get the camera serial number from filename
    """
    parts = filename.split("_")
    sn = parts[2]
    return sn


def get_points(img, file_name, output_dir, pattern_params: Pattern):
    found = False
    corners = 0
    found, corners = cv.findChessboardCorners(img, pattern_params.pattern_size)
    if found:
        print(corners)
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 50, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        frame_img_points = corners.reshape(-1, 2)
        frame_obj_points = pattern_params.pattern_points
    else:
        print("corners not found")
        return None

    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.drawChessboardCorners(vis, pattern_params.pattern_size, corners, found)
    vis = cv.circle(
        vis,
        (int(frame_img_points[0][0]), int(frame_img_points[0][1])),
        10,
        (255, 0, 0),
        2,
    )

    outfile = os.path.join(output_dir, file_name + "_board.png")
    cv.imwrite(outfile, vis)

    print(f"{outfile}... OK")
    return (frame_img_points, frame_obj_points)


# input/output folders
data_dir = "./chessboard_data"
output_dir = "./chessboard_output"

with open("./camera_params.json", "r") as f:
    intrinsic_params = json.load(f)

board_pattern = Pattern(6, 4, 150)


# serial numbers, file names and images
sns_fns_imgs = [
    (
        get_serial_number(file),
        file.split(".png")[0].split("\\")[1],
        cv.imread(file, cv.IMREAD_GRAYSCALE),
    )
    for file in glob(os.path.join(data_dir, "*.png"))
]

extrinsics = []
camera_coords = []


for sn, fn, img in sns_fns_imgs:
    print(f"Processing {fn}...")
    img_points, obj_points = get_points(img, fn, output_dir, board_pattern)
    cameraMatrix = np.array(intrinsic_params[sn]["left_sensor"]["camera_matrix"])
    distCoeffs = np.array(intrinsic_params[sn]["left_sensor"]["distortion_coeff"])
    success, rvec, tvec = cv.solvePnP(
        obj_points,
        img_points,
        cameraMatrix,
        distCoeffs,
        useExtrinsicGuess=False,
        flags=cv.SOLVEPNP_ITERATIVE,
    )  # CV_P3P ,CV_EPNP
    # success, rvec, tvec, inliners = cv.solvePnPRansac(obj_points, img_points,cameraMatrix,distCoeffs)
    if success:
        print(f"{fn}... OK")
        extrinsics.append(
            (sn, fn, np.concatenate((rvec, tvec), 1).reshape(1, 6), rvec, tvec)
        )

        R, jac = cv.Rodrigues(rvec)
        # Xw = -np.matrix(R).T @ tvec
        Xw = -np.matrix(R).T * np.matrix(tvec)
        camera_coords.append(Xw)
        print(f"distance: {round(np.linalg.norm(Xw),2)}")
        print(f"xy dist:  {round(np.linalg.norm(Xw[:2]),2)} \n")

    else:
        print(f"{fn}... FAILED")


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ax.set_aspect("auto")
# ax.set_box_aspect([1, 1, 1])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# ax.set_xlim(-900, 900)
# ax.set_ylim(-900, 900)
# ax.set_zlim(0, 2000)

# Plot axes for reference
ax.quiver(0, 0, 0, 1, 0, 0, color="r", length=100, normalize=True)
ax.quiver(0, 0, 0, 0, 1, 0, color="g", length=100, normalize=True)
ax.quiver(0, 0, 0, 0, 0, 1, color="b", length=100, normalize=True)

# ax.plot([0, 150], [0, 0], zs=[0, 0])


for op in obj_points:
    ax.plot3D(op[0], op[1], op[2], "go")


for coords in camera_coords:
    ax.plot3D(coords[0], coords[1], coords[2], "r^")
    # ax.plot([0, coords[0][0]], [0, coords[1][0]], [0, coords[1][0]], "r--")

plt.show()
