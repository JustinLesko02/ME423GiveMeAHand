### 3D Calibration Updated

import cv2
import numpy as np

# Define checkerboard dimensions (number of inner corners per row and column)
checkerboard_dims = (5, 8)  # Updated dimensions

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points: (0,0,0), (1,0,0), ... for the checkerboard corners
objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images.
objpoints = []      # 3d points in real world space
imgpoints_left = [] # 2d points in left camera image plane.
imgpoints_right = []# 2d points in right camera image plane.

# Activate both cameras (ensure your system assigns the correct indices)
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

print("Press 'c' to capture image pairs for calibration. Press 'q' to quit and calibrate.")

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Failed to grab frames from one or both cameras.")
        break

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Left Camera', frame_left)
    cv2.imshow('Right Camera', frame_right)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Use flags for robust chessboard detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_left_cb, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_dims, flags)
        ret_right_cb, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_dims, flags)

        if ret_left_cb and ret_right_cb:
            # Refine corner locations for more accurate calibration
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners2_left)
            imgpoints_right.append(corners2_right)

            cv2.drawChessboardCorners(frame_left, checkerboard_dims, corners2_left, ret_left_cb)
            cv2.drawChessboardCorners(frame_right, checkerboard_dims, corners2_right, ret_right_cb)
            
            print("Captured image pair for calibration.")
        else:
            print("Chessboard corners not found in both images. Try again.")
            
    elif key == ord('q'):
        break

if len(objpoints) == 0:
    print("No valid image pairs captured. Exiting calibration.")
else:
    # Calibrate each camera individually
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

    # Stereo calibration (fixing intrinsics of each camera)
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        gray_left.shape[::-1], criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
    
    print("Stereo calibration complete.")
    print("Rotation matrix (R):\n", R)
    print("Translation vector (T):\n", T)
    # Optionally, print the RMS reprojection error (retval) to evaluate calibration quality
    print("RMS reprojection error:", retval)

     # Save the calibration results to a file.
    np.savez("stereo_calib.npz", 
             cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
             cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
             R=R, T=T, E=E, F=F, rms_error=retval)
    print("Calibration data saved to stereo_calib.npz")

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
