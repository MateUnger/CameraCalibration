## Charuco_calibration_visualization
Performs extrinsic calibration of 1-4 cameras using a charuco board. Board should be placed in a roughly equidistant manner from all cameras.

Results are saved in 'calibration_results.json' file. When run as a script it also produces an interactive 3D plot of the cameras' positions in relation to the calibraiton pattern. 

Board and calibration battern can be defined, adjusted and generated in using the "generate_charuco_board.py" script. 


## Checkerboard_calibration_visialization
Perform extrinsic calibration of a single or multiple cameras. Board should be positioned as the charuco board. Drawback of this method is the board is not directional meanining, there are multiple solutions for each camera view (pos x and pos x +180 degrees). 