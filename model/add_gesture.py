import cv2
import mediapipe as mp
import csv
import copy
import itertools


NUMBER = 11
csv_path = 'model/new_gestures.csv'
global i
i = 0
global writer

# Define the action to be executed when spacebar is pressed
def execute_action(image, results):
    global i
    global writer

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
            landmark_list = _calc_landmark_list(image, hand_landmarks)
            # print(landmark_list)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = _pre_process_landmark(
                landmark_list)
            
            # print(pre_processed_landmark_list)

            # Write to the dataset file
            # threading.Thread(target=writer.writerow, args=[mode, *pre_processed_landmark_list]).start()
            writer.writerow([NUMBER, *pre_processed_landmark_list])
            print("WRITE " + str(i))
            i += 1


def _calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point



def _pre_process_landmark(landmark_list):
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

# Open the webcam feed
cap = cv2.VideoCapture(0)

with open(csv_path, 'a', newline="") as f:
    writer = csv.writer(f)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    while True:
        # Read a frame from the webcam feed
        ret, frame = cap.read()
        results = None

        with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            # Convert the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)

            # Set the flag to draw the hand landmarks
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)



        # Show the frame in a window named "Webcam Feed"
        cv2.imshow("Webcam Feed", image)

        # Wait for a key to be pressed
        key = cv2.waitKey(1)

        # If the spacebar is pressed, execute the action
        if key == ord(' ') and results.multi_hand_landmarks:
            execute_action(frame, results)

        # If the 'q' key is pressed, exit the loop
        elif key == ord('q'):
            break

    # Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
