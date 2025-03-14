### 3D T7

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
import math

###################Setup for camera#########################
plt.switch_backend('TkAgg')  # Uncomment if needed for an interactive backend

# Load stereo calibration parameters from file
calib_data = np.load("stereo_calib.npz")
cameraMatrix1 = calib_data['cameraMatrix1']
distCoeffs1 = calib_data['distCoeffs1']
cameraMatrix2 = calib_data['cameraMatrix2']
distCoeffs2 = calib_data['distCoeffs2']
R = calib_data['R']
T = calib_data['T']

# Compute projection matrices for left and right cameras.
# Left projection: P1 = K1 * [I | 0]
P1 = cameraMatrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
# Right projection: P2 = K2 * [R | T]
P2 = cameraMatrix2 @ np.hstack((R, T))

# Initialize MediaPipe Pose and Hands for both cameras.
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose_left = mp_pose.Pose()
pose_right = mp_pose.Pose()
hands_left = mp_hands.Hands()
hands_right = mp_hands.Hands()

# Open both camera feeds.
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# Set up a 3D plot with extended limits and adjusted view.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
#ax.set_facecolor('black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=-60)

# Initialize plot objects for joints and links.
joints = ax.scatter([], [], [], c='red', s=50)
lines, = ax.plot([], [], [], '-', color='white', linewidth=2)
finger_lines, = ax.plot([], [], [], '-', color='white', linewidth=2)

last_time = time.time()

###########region: Setup For Arm#########################
##Timing Parameters and methods
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 200
sampleTime = 1/sampleRate

# Load QArm in Position Mode
myArm = QArm(hardware=1)
myArmUtilities = QArmUtilities()
print('Sample Rate is ', sampleRate, ' Hz. Simulation will run until you type Ctrl+C to exit.')

# Reset startTime before Main Loop
startTime = time.time()
#endregion

def check_joint_angle(joint, top_limit, bottom_limit):
    if bottom_limit<joint<top_limit:
        return True
    else:
        return False

def triangulate_points(pt_left, pt_right):
    """
    Given a corresponding point in the left and right images (in pixel coordinates),
    triangulate to obtain its 3D position.
    """
    pts_left = np.array([[pt_left[0]], [pt_left[1]]], dtype=np.float64)
    pts_right = np.array([[pt_right[0]], [pt_right[1]]], dtype=np.float64)
    homog_point = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
    point_3d = homog_point[:3] / homog_point[3]
    return point_3d.flatten()

# def is_fist_closed(triangulated):
#     """Check if the hand is in a fist position based on 3D distances."""
#     if 16 in triangulated and 8 in triangulated:
#         wrist_3d = triangulated[16]
#         index_finger_3d = triangulated[8]
#         elbow_3d = triangulated[14]
#         # Compute Euclidean distance
#         fist_distance = np.linalg.norm(np.array(wrist_3d) - np.array(index_finger_3d))
#         forearm_distance = np.linalg.norm(np.array(elbow_3d) - np.array(wrist_3d))
#         distance = fist_distance/forearm_distance
#         print('fist distance =', distance)
#         return distance < 0.05  # Adjust this threshold as needed
#     return False
# Main update function.

def update(frame):
    global last_time
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        return joints, lines, finger_lines

    # Process both images.
    rgb_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    rgb_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    results_pose_left = pose_left.process(rgb_left)
    results_pose_right = pose_right.process(rgb_right)
    results_hands_left = hands_left.process(rgb_left)
    results_hands_right = hands_right.process(rgb_right)

    # We'll require these pose landmark indices: 12 (shoulder), 14 (elbow), 16 (wrist)
    # And for hand, use index 8 (index finger tip) as representative.
    required_indices = [12, 14, 16, 8]
    
    left_points = {}
    right_points = {}
    # Get pose landmarks from left and right.
    if results_pose_left.pose_landmarks and results_pose_right.pose_landmarks:
        hL, wL, _ = frame_left.shape
        hR, wR, _ = frame_right.shape
        for idx, lm in enumerate(results_pose_left.pose_landmarks.landmark):
            if idx in [12, 14, 16]:
                left_points[idx] = (lm.x * wL, lm.y * hL)
        for idx, lm in enumerate(results_pose_right.pose_landmarks.landmark):
            if idx in [12, 14, 16]:
                right_points[idx] = (lm.x * wR, lm.y * hR)
    # For the hand, use the first detected hand from each side.
    if results_hands_left.multi_hand_landmarks and results_hands_right.multi_hand_landmarks:
        hL, wL, _ = frame_left.shape
        hR, wR, _ = frame_right.shape
        left_hand = results_hands_left.multi_hand_landmarks[0]
        right_hand = results_hands_right.multi_hand_landmarks[0]
        left_points[8] = (left_hand.landmark[8].x * wL, left_hand.landmark[8].y * hL)
        right_points[8] = (right_hand.landmark[8].x * wR, right_hand.landmark[8].y * hR)

    # Check that all required points are detected in both cameras.
    all_detected = True
    missing = []
    for idx in required_indices:
        if idx not in left_points or idx not in right_points:
            all_detected = False
            missing.append(idx)

    triangulated = {}
    if all_detected:
        for idx in required_indices:
            triangulated[idx] = triangulate_points(left_points[idx], right_points[idx])

    # For visualization, we set the shoulder (landmark 12) as the origin.
    joint_x, joint_y, joint_z = [], [], []
    link_x, link_y, link_z = [], [], []
    finger_x, finger_y, finger_z = [], [], []

    shoulder_3d = triangulated.get(12, None)
    elbow_3d = triangulated.get(14, None)
    wrist_3d = triangulated.get(16, None)
    index_3d = triangulated.get(8, None)

    if shoulder_3d is not None:
        origin = shoulder_3d
        def rel(pt):
            return origin-pt
        # Plot joints: shoulder at origin, then elbow, wrist, and index fingertip.
        joint_x.append(0)
        joint_y.append(0)
        joint_z.append(0)
        if elbow_3d is not None:
            r_elbow = rel(elbow_3d)
            joint_x.append(r_elbow[0])
            joint_y.append(r_elbow[1])
            joint_z.append(r_elbow[2])
        if wrist_3d is not None:
            r_wrist = rel(wrist_3d)
            joint_x.append(r_wrist[0])
            joint_y.append(r_wrist[1])
            joint_z.append(r_wrist[2])
        if index_3d is not None:
            r_index = rel(index_3d)
            joint_x.append(r_index[0])
            joint_y.append(r_index[1])
            joint_z.append(r_index[2])
        # Arm links: shoulder→elbow and elbow→wrist.
        if elbow_3d is not None:
            link_x.extend([0, rel(elbow_3d)[0], np.nan])
            link_y.extend([0, rel(elbow_3d)[1], np.nan])
            link_z.extend([0, rel(elbow_3d)[2], np.nan])
        if elbow_3d is not None and wrist_3d is not None:
            link_x.extend([rel(elbow_3d)[0], rel(wrist_3d)[0], np.nan])
            link_y.extend([rel(elbow_3d)[1], rel(wrist_3d)[1], np.nan])
            link_z.extend([rel(elbow_3d)[2], rel(wrist_3d)[2], np.nan])
    
    # Compute robot DOF angles and fist state if all required points exist.
    if all_detected and shoulder_3d is not None and elbow_3d is not None and wrist_3d is not None and index_3d is not None:
        # Define relative positions.
        shoulder_rel = np.array([0, 0, 0])
        elbow_rel = rel(elbow_3d)
        wrist_rel = rel(wrist_3d)

        ### Calculating x,y, and z for the arm
        #print("Wrist Position:", wrist_rel)
        x_arm = wrist_rel[0]
        y_arm = wrist_rel[2]
        z_arm = wrist_rel[1]
        print("arm position", x_arm, y_arm, z_arm)

        x_defaulth = 5  #
        y_defaulth = 1.8  #
        z_defaulth = 2.5  #
        x_defaultQ = 0.45
        y_defaultQ = 0
        z_defaultQ = 0.49
        deltax_h = x_arm - x_defaulth
        deltay_h = y_arm - y_defaulth
        deltaz_h = z_arm - z_defaulth
        xQ = x_defaultQ + (deltax_h)/25  #
        zQ = z_defaultQ + (deltaz_h)/15  #
        yQ = y_defaultQ + (deltay_h)/25  #
        if zQ > 0.55:
            zQ = 0.55
        if zQ < 0.03:
            zQ = 0.03   
        xQ = 0.5
        yQ = 0.5
        zQ = 0.5
        r_cylinder = 0.1475
        r_sphere = 0.8

        ##If the desired arm coordinates are within a cylinder around the shoulder singulatity, saturate them using the slope of the y and x coordinate. 
        if ((xQ**2+yQ**2)<=(r_cylinder**2)) :
            print("Saturation: Shoulder Singularity")
            m = yQ/xQ
            if xQ>=0:
                xQ = r_cylinder/math.sqrt(1+m**2)
                yQ=m*xQ
                if xQ<0:
                    xQ = r_cylinder/(-math.sqrt(1+m**2))
                    yQ = m*xQ
        if ((xQ**2+yQ**2+zQ**2>=r_sphere**2)):
            print("Saturation: Boundary Singularity")
            theta1 = math.atan(yQ/xQ)                                                                                                                                                                                                                  
            theta2 = math.atan(zQ/(math.sqrt(xQ**2+yQ**2)))
            zQ = math.sin(theta2)*r_sphere
            xy = math.cos(theta2)*r_sphere
            yQ = math.sin(theta1)*xy
            xQ = math.cos(theta1)*xy
        
        #print(f"QArm Position x={xQ} y={yQ} z={zQ}")

        index_rel = rel(index_3d)
        # Shoulder yaw: angle between projection of upper arm (shoulder→elbow) on x-z and z-axis.
        v_sh = np.array(elbow_rel)
        shoulder_yaw = np.arctan2(v_sh[0], v_sh[2]) * 180/np.pi
        # Shoulder pitch: angle between v_sh and its x-z projection.
        shoulder_pitch = np.arctan2(v_sh[1], np.sqrt(v_sh[0]**2 + v_sh[2]**2)) * 180/np.pi
        # Elbow flexion: angle between upper arm and forearm.
        v_upper = np.array(elbow_rel)
        v_fore = np.array(wrist_rel) - np.array(elbow_rel)
        norm_upper = np.linalg.norm(v_upper)
        norm_fore = np.linalg.norm(v_fore)
        if norm_upper > 0 and norm_fore > 0:
            elbow_angle = np.arccos(np.clip(np.dot(v_upper, v_fore)/(norm_upper*norm_fore), -1, 1))*180/np.pi
        else:
            elbow_angle = 0.0
        # Wrist rotation: using index fingertip relative to the forearm.
        v_index = np.array(index_rel) - np.array(wrist_rel)
        v_forearm = np.array(wrist_rel) - np.array(elbow_rel)
        norm_forearm = np.linalg.norm(v_forearm)
        if norm_forearm > 0:
            u_forearm = v_forearm / norm_forearm
            v_index_proj = v_index - np.dot(v_index, u_forearm)*u_forearm
            norm_proj = np.linalg.norm(v_index_proj)
            if norm_proj > 0:
                v_index_proj_norm = v_index_proj / norm_proj
                up = np.array([0,1,0])
                ref = up - np.dot(up, u_forearm)*u_forearm
                norm_ref = np.linalg.norm(ref)
                if norm_ref > 0:
                    ref = ref / norm_ref
                    dot_val = np.dot(ref, v_index_proj_norm)
                    cross_val = np.cross(ref, v_index_proj_norm)
                    sign = np.sign(np.dot(cross_val, u_forearm))
                    wrist_angle = sign * np.arccos(np.clip(dot_val, -1, 1)) * 180/np.pi
                else:
                    wrist_angle = 0.0
            else:
                wrist_angle = 0.0
        else:
            wrist_angle = 0.0

        # # Fist flag: here we simply set it True if the index fingertip is very close to the wrist.
        # Function to check if the fist is closed
        def is_fist_closed(triangulated):
            """Check if the hand is in a fist position based on 3D distances."""
            if 16 in triangulated and 8 in triangulated:
                wrist_3d = triangulated[16]
                index_finger_3d = triangulated[8]

                # Compute Euclidean distance between wrist and index fingertip
                fist_distance = np.linalg.norm(np.array(wrist_3d) - np.array(index_finger_3d))
                print("fist distance:", fist_distance)
                # Define a threshold for determining if the hand is closed
                threshold = 5  # Adjust as needed based on real-world calibration
                return fist_distance < threshold  # Returns True if the fist is closed
            return False

        # Fist flag: Determine if the fist is closed
        fist_closed = is_fist_closed(triangulated)
        start = elapsed_time()
        ledCmd = np.array([0, 1, 1], dtype=np.float64)
        result = [xQ, -yQ, zQ, 0.6*fist_closed] # [xQ, yQ, zQ, fist_closed], default [0.45, 0, 0.49, fist_closed]
        print("result", result)
        result = [0.45, 0, 0.49, 0]
        positionCmd = result[0:3]
        gripCmd = result[3]
        if (xQ is not None) and (yQ is not None) and (zQ is not None):
            allPhi, phiCmd = myArmUtilities.qarm_inverse_kinematics(positionCmd, 0, myArm.measJointPosition[0:4])
           # print("phiCmd:", phiCmd)
            if check_joint_angle(phiCmd[0], 160*math.pi/180, -160*math.pi/180) and check_joint_angle(phiCmd[1], 80*math.pi/180, -80*math.pi/180) and check_joint_angle(phiCmd[2], 70*math.pi/2, -90*math.pi/2) and check_joint_angle(phiCmd[3], 160*math.pi/2, -160*math.pi/2):
                myArm.read_write_std(phiCMD=phiCmd, grpCMD=gripCmd, baseLED=ledCmd)
                #print ("x_arm", x_arm, "z arm", z_arm)
            else:
                print("Saturation: Joint Limit")
                
        # if time.time() - last_time >= 0.5:
        #     print(f"Shoulder Yaw: {shoulder_yaw:.2f}°, Shoulder Pitch: {shoulder_pitch:.2f}°, "
        #           f"Elbow: {elbow_angle:.2f}°, Wrist: {wrist_angle:.2f}°")
        #     print(f"Fist Closed: {fist_closed}")
        #     last_time = time.time()


    else:
        if time.time() - last_time >= 0.5:
            print("Missing required landmarks for triangulation:", missing)
            last_time = time.time()

    # Update the 3D plot.
    joints._offsets3d = (np.array(joint_x), np.array(joint_y), np.array(joint_z))
    lines.set_data(link_x, link_y)
    lines.set_3d_properties(link_z)
    finger_lines.set_data(finger_x, finger_y)
    finger_lines.set_3d_properties(finger_z)
    
    
    last_time = time.time()
    # Overlay status on the camera feeds.
    status_text = "All joints detected" if all_detected else f"Missing: {missing}"
    cv2.putText(frame_left, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if all_detected else (0, 0, 255), 2)
    cv2.imshow("Left Camera Feed", frame_left)
    cv2.putText(frame_right, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if all_detected else (0, 0, 255), 2)
    cv2.imshow("Right Camera Feed", frame_right)
    cv2.waitKey(1)

    return joints, lines, finger_lines
try:
    while myArm.status:
        ani = FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
        plt.show()
        plt.flush()
except KeyboardInterrupt:
    print("User interrupted!")
finally:
    myArm.terminate()
    # Show plot
    # Cleanup
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


