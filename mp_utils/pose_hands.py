from mp_utils import mp_hands
from mp_utils import mp_pose

class HandPoseDetection():

    def __init__(self,
                hand_region_window_width=300,
                hand_region_window_height=300,
                pose_model_asset_path='model/pose_landmarker_heavy.task',
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_pose_tracking_confidence=0.3,
                safe_zone = True,
                hand_model_asset_path='model/hand_landmarker.task',
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_hand_tracking_confidence=0.3
                ):
        self.pose = mp_pose.PoseDetection(
                hand_region_window_width=hand_region_window_width,
                hand_region_window_height=hand_region_window_height,
                model_asset_path=pose_model_asset_path,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_pose_tracking_confidence,
                safe_zone = safe_zone
                )
        self.hands = mp_hands.HandDetection(
                model_asset_path=hand_model_asset_path,
                min_hand_detection_confidence=min_hand_detection_confidence,
                min_hand_presence_confidence=min_hand_presence_confidence,
                min_tracking_confidence=min_hand_tracking_confidence
                )

    def close(self):
        self.pose.close()
        self.hands.close()

    def extract_pose(self, image):
        return self.pose.extract_pose(image)

    def extract_hands(self, image):
        return self.hands.extract_hands(image)

    def draw_hands(self, image):
        return self.hands.draw_hands(image)
    
    def draw_pose(self, image):
        return self.pose.draw_pose(image)

    def extract_right_hand_roi(self, image):
        return self.pose.extract_right_hand_region(image)