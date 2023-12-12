import cv2
import mediapipe as mp
import csv
import copy
import itertools
import time
import numpy as np

from mediapipe.framework.formats import landmark_pb2

NUMBER = 2
i = 0

class landmarker_and_result():

    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()
    
    def createLandmarker(self):
        # callback function
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path="model/hand_landmarker.task"), # path to model
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
            num_hands = 1, # track both hands
            min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
            min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
            min_tracking_confidence = 0.3, # lower than value to get predictions more often
            result_callback=update_result)
        
        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)
    
    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        # close landmarker
        self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
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

def add_to_csv(image, results, writer):
    global i
    if results.hand_landmarks is not None:
        for hand_landmarks in results.hand_landmarks:
            landmark_list = calc_landmark_list(image, hand_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            writer.writerow([NUMBER, *pre_processed_landmark_list])
            print("WRITE " + str(i))
            i += 1

def calc_landmark_list(image, landmarks):
    # image_width, image_height = image.shape[1], image.shape[0]
    image_width, image_height = 300, 300

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list
cap = cv2.VideoCapture(0)

# create landmarker
hand_landmarker = landmarker_and_result()
csv_path = 'model/new_gestures.csv'

with open(csv_path, 'a', newline="") as f:
    writer = csv.writer(f)
    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        # frame_resized = cv2.resize(frame, (width/2, height/2))
        min_dimension = min(height, width)
        start_x = int((width - min_dimension) / 2)
        start_y = int((height - min_dimension) / 2)

        # Crop the frame to a square
        square_frame = frame[start_y:start_y + min_dimension, start_x:start_x + min_dimension]
        square_frame = np.array(square_frame)
        # square_frame = frame

        # update landmarker results
        hand_landmarker.detect_async(square_frame)
        # draw landmarks on frame
        square_frame = draw_landmarks_on_image(square_frame,hand_landmarker.result)

        # display image
        cv2.putText(square_frame, f'{NUMBER}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv2.imshow('frame',square_frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord(' '):
            add_to_csv(square_frame, hand_landmarker.result, writer)
        elif key == ord('n'):
            NUMBER += 1
            i = 0
        elif key == ord('p'):
            NUMBER -= 1
            i = 0
    
hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()