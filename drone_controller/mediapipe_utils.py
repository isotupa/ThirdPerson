import mediapipe as mp
import numpy as np
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

constant_width = 300
constant_height = 300

global pose
global hands

def extract_pose_and_hands(image):
    original = image.copy()
    pose_results = extract_pose(image)
    if pose_results.pose_landmarks:
        right_hand_roi = extract_hand_region(original, pose_results)
        if right_hand_roi is not None:
            hands_results = extract_hands(right_hand_roi)
            return hands_results, right_hand_roi
    return None, None

def initialise_pose(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    global pose
    pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)
    
def initialise_hands(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    global hands
    hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence) 

def terminate_pose():
    pose.close()

def terminate_hands():
    hands.close()

def extract_hands(image):
    hands_result = hands.process(image)
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return hands_result

# Image must be RGB for optimal detection
def extract_pose(image):
    pose_results = pose.process(image)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return pose_results

def extract_hand_region(image, pose_results):

    left_hand_indices = [15, 17, 19, 21]
    right_hand_indices = [16, 18, 20, 22]

    # Convert the landmarks to pixel coordinates
    image_height, image_width, _ = image.shape
    left_hand_pixels = [(pose_results.pose_landmarks.landmark[index].x * image_width,
                            pose_results.pose_landmarks.landmark[index].y * image_height) for index in
                        left_hand_indices]
    right_hand_pixels = [(pose_results.pose_landmarks.landmark[index].x * image_width,
                            pose_results.pose_landmarks.landmark[index].y * image_height) for index in
                            right_hand_indices]

    # Calculate distances between specific landmarks on the hand
    left_distance = np.linalg.norm(np.array(left_hand_pixels[1]) - np.array(left_hand_pixels[0]))
    right_distance = np.linalg.norm(np.array(right_hand_pixels[1]) - np.array(right_hand_pixels[0]))

    # Modify rectangle size based on distances
    scale_factor = 5.0

    # Adjust rectangle size proportionally to the distance
    left_hand_w = int(left_distance * scale_factor)
    left_hand_h = int(left_distance * scale_factor)
    right_hand_w = int(right_distance * scale_factor)
    right_hand_h = int(right_distance * scale_factor)

    # Get the centroid of the left and right hands
    left_hand_centroid = np.mean(left_hand_pixels, axis=0, dtype=np.float32)
    right_hand_centroid = np.mean(right_hand_pixels, axis=0, dtype=np.float32)

    # Ensure the coordinates are within the image boundaries
    left_hand_x = max(0, int(left_hand_centroid[0] - left_hand_w // 2))
    left_hand_y = max(0, int(left_hand_centroid[1] - left_hand_h // 2))
    right_hand_x = max(0, int(right_hand_centroid[0] - right_hand_w // 2))
    right_hand_y = max(0, int(right_hand_centroid[1] - right_hand_h // 2))

    # Extract regions within the rectangles
    left_hand_region = image[left_hand_y:left_hand_y + left_hand_h, left_hand_x:left_hand_x + left_hand_w]
    right_hand_region = image[right_hand_y:right_hand_y + right_hand_h, right_hand_x:right_hand_x + right_hand_w]

    cv.rectangle(image, (left_hand_x, left_hand_y), (left_hand_x + left_hand_w, left_hand_y + left_hand_h),
                    (0, 255, 0), 2)
    cv.rectangle(image, (right_hand_x, right_hand_y),
                    (right_hand_x + right_hand_w, right_hand_y + right_hand_h), (0, 255, 0), 2)

    elbow_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    wrist_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Extract y-coordinates of elbow and wrist
    elbow_y = int(elbow_landmark.y * image_height)
    wrist_y = int(wrist_landmark.y * image_height)
    
    # Display right hand pixels in a new window
    if right_hand_region.shape[0] > 0 and right_hand_region.shape[1] > 0 and wrist_y < elbow_y:
        # Display right hand pixels in a new window
        right_hand_region_resized = cv.resize(right_hand_region, (constant_width, constant_height))
        return right_hand_region_resized
    return None
