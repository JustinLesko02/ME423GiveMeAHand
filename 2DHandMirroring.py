import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
import time
import math
###################Setup for camera#########################
plt.switch_backend('TkAgg')  # Uncomment if needed for an interactive backend

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Open camera feed
# # Try camera indices 0 to 3 (adjust range as needed)
# for index in range(4):
#     cap = cv2.VideoCapture(index)
#     if cap.isOpened():
#         print(f"Camera index {index} is working.")
#     else:
#         print(f"Camera index {index} is NOT working.")
#     cap.release()

cap = cv2.VideoCapture(2)


# Frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Landmarks for right shoulder, elbow, wrist, and fingers
selected_indices = [12, 14, 16]  # Shoulder, elbow, wrist
finger_indices = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky

# Connections for drawing
link_pairs = [(12, 14), (14, 16)]  # Shoulder to elbow, elbow to wrist
finger_links = [(16, 4), (16, 8), (16, 12), (16, 16), (16, 20)]  # Wrist to fingers

# Initialize plot
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_facecolor('black')

# Scatter and line objects
joints, = ax.plot([], [], 'o', color='red', markersize=5)
lines, = ax.plot([], [], '-', color='white', linewidth=2)
finger_lines, = ax.plot([], [], '-', color='white', linewidth=2)

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
def calculate_angles(shoulder, elbow, wrist):
    """Calculate angles for upper arm and forearm."""
    if shoulder is None or elbow is None or wrist is None:
        return None, None, None, None
    # find default config
    x = wrist[0]
    y = wrist[1]
    print(f"x = {x} y = {y}")
    vec1 = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    vec2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
    x_default = 0.3
    y_default = 0.3
    delta_x = wrist[0] - x_default
    delta_y = wrist[1] - y_default
    x_Qarm_default = 0.45 
    z_Qarm_default = 0.49
    scalar = 0.75
    x_arm = x_Qarm_default + scalar*delta_x*(1)
    z_arm = z_Qarm_default + scalar*delta_y*(1)
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])
    theta1 = np.degrees(angle1)
    theta2 = np.degrees(angle2 - angle1)
    # Start timing this iteration
    return theta1, theta2, x_arm, z_arm

def is_fist_closed(hand_points):
    """Check if the hand is in a fist position based on finger distances."""
    if hand_points:
        distances = [np.linalg.norm(np.array(hand_points[16]) - np.array(hand_points[idx])) 
                     for idx in finger_indices if idx in hand_points]
        return np.mean(distances) < 0.03  # Threshold for fist detection
    return False

def normalize_coord(x, y, shoulder_x, shoulder_y):
    """Normalize coordinates relative to the right shoulder."""
    x_norm = (shoulder_x - x) / frame_width
    y_norm = (shoulder_y - y) / frame_height
    return x_norm, y_norm

def check_joint_angle(joint, top_limit, bottom_limit):
    if bottom_limit<joint<top_limit:
        return True
    else:
        return False
    
def update(frame):
    """Update function for animation."""
    global last_time
    ret, frame = cap.read()
    if not ret:
        return joints, lines, finger_lines

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    joint_x, joint_y = [], []
    link_x, link_y = [], []
    finger_x, finger_y = [], []
    hand_points = {}

    shoulder, elbow, wrist = None, None, None

    # Process pose landmarks
    if results_pose.pose_landmarks:
        landmarks = {}
        shoulder_x, shoulder_y = None, None

        for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
            x, y = int(lm.x * frame_width), int(lm.y * frame_height)

            if idx == 12:  # Shoulder (reference)
                shoulder_x, shoulder_y = x, y
                landmarks[idx] = (0, 0)
                joint_x.append(0)
                joint_y.append(0)
                shoulder = (0, 0)

            elif idx in selected_indices and shoulder_x is not None:
                x_norm, y_norm = normalize_coord(x, y, shoulder_x, shoulder_y)
                landmarks[idx] = (x_norm, y_norm)
                joint_x.append(x_norm)
                joint_y.append(y_norm)

                if idx == 14:
                    elbow = (x_norm, y_norm)
                elif idx == 16:
                    wrist = (x_norm, y_norm)

        # Link arm joints
        for start_idx, end_idx in link_pairs:
            if start_idx in landmarks and end_idx in landmarks:
                link_x.extend([landmarks[start_idx][0], landmarks[end_idx][0], None])
                link_y.extend([landmarks[start_idx][1], landmarks[end_idx][1], None])

    # Process hand landmarks
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for idx in finger_indices:
                lm = hand_landmarks.landmark[idx]
                x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                if shoulder_x is not None:
                    x_norm, y_norm = normalize_coord(x, y, shoulder_x, shoulder_y)
                    hand_points[idx] = (x_norm, y_norm)
                    joint_x.append(x_norm)
                    joint_y.append(y_norm)

            # Connect wrist to fingers
            if wrist:
                for start_idx, end_idx in finger_links:
                    if end_idx in hand_points:
                        finger_x.extend([wrist[0], hand_points[end_idx][0], None])
                        finger_y.extend([wrist[1], hand_points[end_idx][1], None])

    # Calculate angles and check for fist closure every 0.5s
    if time.time() - last_time >= 0.5:
        theta1, theta2, x_arm, z_arm = calculate_angles(shoulder, elbow, wrist) #Get thetas and coordinates of x and z of wrist
        fist_closed = 0.6*is_fist_closed(hand_points)
        start = elapsed_time()
        ledCmd = np.array([0, 1, 1], dtype=np.float64)
        result =[x_arm, 0, z_arm, fist_closed] #[x_arm, 0, z_arm, fist_closed] # default [0.45, 0, 0.49, fist_closed]
        positionCmd = result[0:3]
        gripCmd = result[3]
        if (x_arm is not None) and (z_arm is not None):
            allPhi, phiCmd = myArmUtilities.qarm_inverse_kinematics(positionCmd, 0, myArm.measJointPosition[0:4])
            print("phiCmd:", phiCmd)
            if check_joint_angle(phiCmd[0], 160*math.pi/180, -160*math.pi/180) and check_joint_angle(phiCmd[1], 80*math.pi/180, -80*math.pi/180) and check_joint_angle(phiCmd[2], 70*math.pi/2, -90*math.pi/2) and check_joint_angle(phiCmd[3], 160*math.pi/2, -160*math.pi/2):
                myArm.read_write_std(phiCMD=phiCmd, grpCMD=gripCmd, baseLED=ledCmd)
                print ("x_arm", x_arm, "z arm", z_arm)
            else:
                print("Saturation Triggered")
        if theta1 is not None and theta2 is not None:
            print(f"Theta1: {theta1:.2f} degrees, Theta2: {theta2:.2f} degrees")
        print(f"Fist closed: {fist_closed}")
        last_time = time.time()

    # Update plot data
    joints.set_data(joint_x, joint_y)
    lines.set_data(link_x, link_y)
    finger_lines.set_data(finger_x, finger_y)

    return joints, lines, finger_lines

# Create animation
try:
    while myArm.status:
        ani = FuncAnimation(fig, update, interval=33, blit=True, cache_frame_data=False)  # 30 FPS (1000ms / 30 ≈ 33ms)
        plt.show()
        plt.flush()
        
except KeyboardInterrupt:
    print("User interrupted!")
finally:
    myArm.terminate()
    # Show plot
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
