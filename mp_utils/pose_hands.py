from mp_utils import mp_hands
from mp_utils import mp_pose

class HandPoseDetection():

    def __init__(self):
        self.pose = mp_pose.PoseDetection()
        self.hands = mp_hands.HandDetection()

    def close(self):
        self.pose.close()
        self.hands.close()

    def extract_pose_and_hands(self, image):
        original = image.copy()
        pose_results = self.pose.extract_pose(image)
        if pose_results.pose_landmarks:
            right_hand_roi = self.pose.extract_right_hand_region(original, pose_results)
            if right_hand_roi is not None:
                hands_results = self.hands.extract_hands(right_hand_roi)
                return pose_results, hands_results, right_hand_roi
            return pose_results, None, None
        return None, None, None

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