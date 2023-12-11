import mediapipe as mp
import numpy as np
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

constant_width = 300
constant_height = 300

safe_zone = False

global pose
global hands

def extract_pose_and_hands(image):
    original = image.copy()
    pose_results = extract_pose(image)
    if pose_results.pose_landmarks:
        right_hand_roi = extract_hand_region(original, pose_results)
        if right_hand_roi is not None:
            hands_results = extract_hands(right_hand_roi)
            return pose_results, hands_results, right_hand_roi
        return pose_results, None, None
    return None, None, None

def initialise_pose(min_detection_confidence=0.6, min_tracking_confidence=0.3):
    global pose
    pose = mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)
    
def initialise_hands(min_detection_confidence=0.2, min_tracking_confidence=0.2):
    global hands
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    hands = vision.HandLandmarker.create_from_options(options)
    # hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence,
    #                      min_tracking_confidence=min_tracking_confidence) 

def terminate_pose():
    pose.close()

def terminate_hands():
    hands.close()

def extract_hands(image):
    if image is None:
        return None
    # hands_result = hands.process(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    hands_result = hands.detect(mp_image)
    return hands_result
    
def draw_hands(image, hands_result):
    # if image is None or hands_result is None:
    if image is None:
        return None
    return draw_landmarks_on_image(image, hands_result)
    # if hands_result.multi_hand_landmarks:
    #     for hand_landmarks in hands_result.multi_hand_landmarks:
    #         mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image

def extract_pose(image):
    if image is not None:
        pose_results = pose.process(image)
        return pose_results
    return None

def draw_pose(image, pose_results):
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def extract_hand_region(image, pose_results):
    if image is None:
        return None
    if pose_results is None or pose_results.pose_landmarks is None:
        return np.zeros((constant_height, constant_width, 3), np.uint8)

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
    # left_hand_region = image[left_hand_y:left_hand_y + left_hand_h, left_hand_x:left_hand_x + left_hand_w]
    right_hand_region = image.copy()[right_hand_y:right_hand_y + right_hand_h, right_hand_x:right_hand_x + right_hand_w]

    # cv.rectangle(image, (left_hand_x, left_hand_y), (left_hand_x + left_hand_w, left_hand_y + left_hand_h),
    #                 (0, 255, 0), 2)
    cv.rectangle(image, (right_hand_x, right_hand_y),
                    (right_hand_x + right_hand_w, right_hand_y + right_hand_h), (0, 255, 0), 2)

    elbow_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    wrist_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Extract y-coordinates of elbow and wrist
    elbow_y = int(elbow_landmark.y * image_height)
    wrist_y = int(wrist_landmark.y * image_height)
    
    # Display right hand pixels in a new window
    if right_hand_region.shape[0] > 0 and right_hand_region.shape[1] > 0:
        # Display right hand pixels in a new windo
        if wrist_y < elbow_y or safe_zone:
            right_hand_region_resized = cv.resize(right_hand_region, (constant_width, constant_height))
            return right_hand_region_resized

    return np.zeros((constant_height, constant_width, 3), np.uint8)
