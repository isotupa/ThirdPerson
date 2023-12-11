import mediapipe as mp
import numpy as np
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

constant_width = 300
constant_height = 300

safe_zone = False

global pose
global hands

class HandPoseDetection:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = None
        self.hands = None
        self.pose_result = mp.tasks.vision.PoseLandmarkerResult
        self.constant_width = 300
        self.constant_height = 300
        self.safe_zone = False
        self.hand_result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.initialise_hands()
        self.initialise_pose()

    
    def extract_pose_and_hands(self, image):
        original = image.copy()
        pose_results = self.extract_pose(image)
        if pose_results.pose_landmarks:
            right_hand_roi = self.extract_hand_region(original, pose_results)
            if right_hand_roi is not None:
                hands_results = self.extract_hands(right_hand_roi)
                return pose_results, hands_results, right_hand_roi
            return pose_results, None, None
        return None, None, None

    def initialise_pose(self, min_detection_confidence=0.6, min_tracking_confidence=0.3):
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.pose_result = result

        base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
                    num_poses = 1,
                    min_pose_detection_confidence = 0.3,
                    min_pose_presence_confidence = 0.3,
                    min_tracking_confidence = 0.3,
                    result_callback=update_result
                    )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        
    def initialise_hands(self, min_detection_confidence=0.2, min_tracking_confidence=0.2):

        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.hand_result = result

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
                    num_hands = 1,
                    min_hand_detection_confidence = 0.3,
                    min_hand_presence_confidence = 0.3,
                    min_tracking_confidence = 0.3,
                    result_callback=update_result
                    )
        self.hands = vision.HandLandmarker.create_from_options(options)
        # hands = mp_hands.Hands(min_detection_confidence=min_detection_confidence,
        #                      min_tracking_confidence=min_tracking_confidence) 
    
    def terminate_pose(self):
        self.pose.close()

    def terminate_hands(self):
        self.hands.close()

    def extract_hands(self, image):
        if image is None:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.hands.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
        # hands_result = hands.process(image)
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # hands_result = self.hands.detect(mp_image)
        return self.hand_result
        # return hands_result
        
    def draw_hands(self, image, hands_result):
        # if image is None or hands_result is None:
        if image is None:
            return None
        return self.draw_landmarks_on_image(image, hands_result)

    def draw_landmarks_on_image(self, rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
        try:
            if detection_result.hand_landmarks == []:
                return rgb_image
            else:
                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness
                annotated_image = np.copy(rgb_image)

                # Loop through the detected hands to visualize.
                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]
                    
                    # Draw the hand landmarks.
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
                    mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                return annotated_image
        except:
            return rgb_image

    def extract_pose(self, image):
        if image is None:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.pose.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
        return self.pose_result
        # if image is not None:
        #     pose_results = self.pose.process(image)
        #     return pose_results
        # return None

    def draw_pose(self, rgb_image, detection_result):
        # if pose_results.pose_landmarks:
        try:
            pose_landmarks_list = detection_result.pose_landmarks
            annotated_image = np.copy(rgb_image)

            # Loop through the detected poses to visualize.
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]

                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )
            return annotated_image       
        except:
            return rgb_image
     #     mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    def extract_hand_region(self, image, pose_results):
        if image is None:
            return None
        if pose_results is None or not hasattr(pose_results, 'pose_landmarks') or len(pose_results.pose_landmarks) == 0:
            return np.zeros((constant_height, constant_width, 3), np.uint8)

        left_hand_indices = [15, 17, 19, 21]
        right_hand_indices = [16, 18, 20, 22]

        # Convert the landmarks to pixel coordinates
        image_height, image_width, _ = image.shape
        # left_hand_pixels = [(pose_results.pose_landmarks[index].x * image_width,
        #                         pose_results.pose_landmarks[index].y * image_height) for index in
        #                     left_hand_indices]
        pose_landmark_list = pose_results.pose_landmarks
        pose_landmarks = pose_landmark_list[0]
        right_hand_pixels = [(pose_landmarks[index].x * image_width,
                                pose_landmarks[index].y * image_height) for index in
                                right_hand_indices]

        # Calculate distances between specific landmarks on the hand
        # left_distance = np.linalg.norm(np.array(left_hand_pixels[1]) - np.array(left_hand_pixels[0]))
        right_distance = np.linalg.norm(np.array(right_hand_pixels[1]) - np.array(right_hand_pixels[0]))

        # Modify rectangle size based on distances
        scale_factor = 5.0

        # Adjust rectangle size proportionally to the distance
        # left_hand_w = int(left_distance * scale_factor)
        # left_hand_h = int(left_distance * scale_factor)
        right_hand_w = int(right_distance * scale_factor)
        right_hand_h = int(right_distance * scale_factor)

        # Get the centroid of the left and right hands
        # left_hand_centroid = np.mean(left_hand_pixels, axis=0, dtype=np.float32)
        right_hand_centroid = np.mean(right_hand_pixels, axis=0, dtype=np.float32)

        # Ensure the coordinates are within the image boundaries
        # left_hand_x = max(0, int(left_hand_centroid[0] - left_hand_w // 2))
        # left_hand_y = max(0, int(left_hand_centroid[1] - left_hand_h // 2))
        right_hand_x = max(0, int(right_hand_centroid[0] - right_hand_w // 2))
        right_hand_y = max(0, int(right_hand_centroid[1] - right_hand_h // 2))

        # Extract regions within the rectangles
        # left_hand_region = image[left_hand_y:left_hand_y + left_hand_h, left_hand_x:left_hand_x + left_hand_w]
        right_hand_region = image.copy()[right_hand_y:right_hand_y + right_hand_h, right_hand_x:right_hand_x + right_hand_w]

        # cv.rectangle(image, (left_hand_x, left_hand_y), (left_hand_x + left_hand_w, left_hand_y + left_hand_h),
        #                 (0, 255, 0), 2)
        cv.rectangle(image, (right_hand_x, right_hand_y),
                        (right_hand_x + right_hand_w, right_hand_y + right_hand_h), (0, 255, 0), 2)

        elbow_landmark = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist_landmark = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

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
