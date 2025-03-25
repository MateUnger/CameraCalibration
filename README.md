# Charuco_calibration_visualization
Performs extrinsic calibration of 1-4 cameras using a charuco board. Board should be placed in a roughly equidistant manner from all cameras.

Results are saved in 'calibration_results.json' file. When run as a script it also produces an interactive 3D plot of the cameras' positions in relation to the calibraiton pattern. 

[OpenCV Pose Overview](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)

[Charuco Documentation](https://docs.opencv.org/3.4/d9/d6a/group__aruco.html)




## Coordinate transforms used

### Camera coord system --> World coord system

P<sub>camera</sub> = R @ P<sub>world</sub> + tvec

where:
* P<sub>camera</sub> = [Cx, Cy, Cz] is a point defined in the camera's coordinate system
* P<sub>world</sub> = [Wx, Wy, Wz] is a point defined in the global / world coordinate system 
* R is the rotation matrix (cv.Rodrigues(rvec))
* tvec is the translation vector


### World coord system --> Camera coord system


P<sub>world</sub> = R<sup>-1</sup> @ (P<sub>camera</sub> - tvec)  =  R<sup>T</sup> @ (P<sub>camera</sub> - tvec)

if we want to know the camera's position (P<sub>camera</sub> = [0, 0, 0]) in world coordinates this becomes:

P<sub>world</sub> = -R<sup>T</sup> @ tvec


___

## ChArUco board 
Board and calibration pattern can be defined, adjusted and generated in using the "generate_charuco_board.py" script. Marker and checkerboard parameters either have to fit a predefined dictionary (opencv), or a custom dict could be generated (not implemented here yet). 


___
___

# Checkerboard_calibration_visialization
Perform extrinsic calibration of a single or multiple cameras. Board should be positioned as the charuco board. Drawback of this method is the board is not directional meanining, there are multiple solutions for each camera view (pos x and pos x +180 degrees). 