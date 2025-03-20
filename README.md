# ME423GiveMeAHand
This repository contains code that manipulates a Quansar QArm using hand motions captured on a webcam.
## Overview
2dhandmirroring.py - 
3dHandMirroring.py - 
cameracalibration.py - 
## Necessary equipment
 - A computer or microprocessor
 - One or two cameras (for 2d and 3d manipulation respectively)
 - One Quansar QArm
## Operation
In order to run the 2d code, press "run" and sit in front of the camera with your hand up (like you are waving). After a long windup, the code should start, showing a live capture of the wrist position in a figure. Additional calibration may be needed for default hand position and scaling between hand movement and robotic hand movement. 
In order to run the 3d code, first set up both cameras and run cameracalibration.py. This will produce a stereo data processing file containing the transformation of one camera to the next. In the same folder, then run the 3d hand mirroring code. Once again, there will be a long windup, then both cameras should turn on. Make sure your arm is centered in both cameras, and perform a similar adjustment process as for 2d - capture your hand position and desired movement sensitivity and put them in the defaulting code. 
