from drone_controller import drone
from drone_controller import mediapipe_utils
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
import cv2 as cv
import numpy as np


def main():
    # Connect to the drone
    # drone_instance = drone.connect_to_drone()  # Function in drone.py
    cap = cv.VideoCapture(0)

    mediapipe_utils.initialise_hands()
    mediapipe_utils.initialise_pose()

    instructions = gesture_instructions.Instructions()
    buffer = gesture_buffer.GestureBuffer()
    
    while True:
        # Read image from drone camera
        # image = drone.get_camera_image(drone_instance)  # Function in drone.py
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        ret, image = cap.read()

        gesture = -1
        
        # Extract pose and hands using MediaPipe
        pose, hands, right_hand_roi = mediapipe_utils.extract_pose_and_hands(image)
        if hands:
            # Recognize gesture using neural network
            gesture, labels = gesture_recognition.recognize_gesture(hands, right_hand_roi)
            gesture_name = gesture_recognition.translate_gesture_id_to_name(gesture, labels)
            buffer.add_gesture(gesture)
            gesture = buffer.get_gesture()
            
            cv.putText(right_hand_roi, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (150,0,0), 2)

            cv.imshow('Right Hand', right_hand_roi)
        
        # Map recognized gesture to instructions
        move = instructions.calculate_move(gesture, pose, image)
        print(move)
        
        # Execute instruction
        drone.execute_instruction(move)  # Function in drone.py
        # cv.putText(image, move, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.putText(image, f'{instructions.get_follow_state()}', (330, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.imshow('ThirdPerson', image)
        
        key = cv.waitKey(1)
        if key == ord('q'):
            # drone.terminate_drone(drone_instance)
            mediapipe_utils.terminate_hands()
            mediapipe_utils.terminate_pose()
            break
        elif key == ord(' '):
            # drone.initialise_drone(drone_instance)
            pass

if __name__ == "__main__":
    main()