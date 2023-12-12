import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import time
import numpy as np
import cv2 as cv

class PoseDetection():

    def __init__(self,
                 hand_region_window_width=300,
                 hand_region_window_height=300,
                 model_asset_path='model/pose_landmarker_heavy.task',
                 min_pose_detection_confidence=0.3,
                 min_pose_presence_confidence=0.3,
                 min_tracking_confidence=0.3,
                 safe_zone = True
                 ) -> None:
        self.hand_region_window_width = hand_region_window_width
        self.hand_region_window_height = hand_region_window_height
        self.pose_result = mp.tasks.vision.PoseLandmarkerResult
        self.landmarker = mp.tasks.vision.PoseLandmarker
        self.pose = None
        self.initialise_pose(model_asset_path, 
                             min_pose_detection_confidence, 
                             min_pose_presence_confidence,
                             min_tracking_confidence)
        self.safe_zone = safe_zone

    def initialise_pose(self, model_asset_path, 
                             min_pose_detection_confidence, 
                             min_pose_presence_confidence,
                             min_tracking_confidence):

        def update_result(result: mp.tasks.vision.HandLandmarkerResult, 
                          output_image: mp.Image, timestamp_ms: int):
            self.pose_result = result

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
                    num_poses = 1,
                    min_pose_detection_confidence = min_pose_detection_confidence,
                    min_pose_presence_confidence = min_pose_presence_confidence,
                    min_tracking_confidence = min_tracking_confidence,
                    result_callback=update_result
                    )
        self.pose = vision.PoseLandmarker.create_from_options(options)
    
    def close(self):
        self.pose.close()
    
    def extract_pose(self, image):
        if image is None:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.pose.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
        return self.pose_result
    
    def draw_pose(self, rgb_image):
        # if pose_results.pose_landmarks:
        try:
            detection_result = self.pose_result
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


    def extract_right_hand_region(self, image):
        if image is None:
            return None
        pose_results = self.pose_result
        if pose_results is None or not hasattr(pose_results, 'pose_landmarks') or len(pose_results.pose_landmarks) == 0:
            return np.zeros((self.hand_region_window_height, self.hand_region_window_width, 3), np.uint8)

        right_hand_indices = [15, 17, 19, 21]
        # right_hand_indices = [16, 18, 20, 22]

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
        original = image.copy()
        right_hand_region = original[right_hand_y:right_hand_y + right_hand_h, right_hand_x:right_hand_x + right_hand_w]

        # cv.rectangle(image, (left_hand_x, left_hand_y), (left_hand_x + left_hand_w, left_hand_y + left_hand_h),
        #                 (0, 255, 0), 2)

        cv.rectangle(image, (right_hand_x, right_hand_y),
                        (right_hand_x + right_hand_w, right_hand_y + right_hand_h), (0, 255, 0), 2)

        elbow_landmark = pose_landmarks[solutions.pose.PoseLandmark.LEFT_ELBOW]
        wrist_landmark = pose_landmarks[solutions.pose.PoseLandmark.LEFT_WRIST]

        # Extract y-coordinates of elbow and wrist
        elbow_y = int(elbow_landmark.y * image_height)
        wrist_y = int(wrist_landmark.y * image_height)
        
        # Display right hand pixels in a new window
        if right_hand_region.shape[0] > 0 and right_hand_region.shape[1] > 0:
            # Display right hand pixels in a new windo
            if wrist_y < elbow_y or not self.safe_zone:
                right_hand_region_resized = cv.resize(right_hand_region, (self.hand_region_window_width, self.hand_region_window_height))
                return right_hand_region_resized

        return np.zeros((self.hand_region_window_height, self.hand_region_window_width, 3), np.uint8)