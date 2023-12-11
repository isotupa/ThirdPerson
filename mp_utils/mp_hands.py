import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import time
import numpy as np

class HandDetection():

    def __init__(self, model_asset_path='model/hand_landmarker.task',
                 min_hand_detection_confidence=0.3,
                 min_hand_presence_confidence=0.3,
                 min_tracking_confidence=0.3) -> None:
        self.hand_result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.hands = None
        self.initialise_hands(model_asset_path=model_asset_path,
                              min_hand_detection_confidence=min_hand_detection_confidence,
                              min_hand_presence_confidence=min_hand_presence_confidence,
                              min_tracking_confidence=min_tracking_confidence)

    
    def initialise_hands(self, model_asset_path='model/hand_landmarker.task',
                 min_hand_detection_confidence=0.3,
                 min_hand_presence_confidence=0.3,
                 min_tracking_confidence=0.3):

        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.hand_result = result

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
                    num_hands = 1,
                    min_hand_detection_confidence = min_hand_detection_confidence,
                    min_hand_presence_confidence = min_hand_presence_confidence,
                    min_tracking_confidence = min_tracking_confidence,
                    result_callback=update_result
                    )
        self.hands = vision.HandLandmarker.create_from_options(options)

    def close(self):
        self.hands.close()

    def extract_hands(self, image):
        if image is None:
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        self.hands.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

        return self.hand_result

    def draw_hands(self, rgb_image):
        if rgb_image is None:
            return rgb_image
        try:
            hands_result = self.hand_result
            if hands_result.hand_landmarks == []:
                return rgb_image
            else:
                hand_landmarks_list = hands_result.hand_landmarks
                handedness_list = hands_result.handedness
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



