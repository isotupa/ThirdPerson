# from drone_controller import drone
from drone_controller import mediapipe_utils
from webcam_sim import webcam_functions
from neural_network import gesture_recognition
# from instructions import gesture_instructions
import cv2 as cv

def main():
    # Connect to the drone
    # drone_instance = drone.connect_to_drone()  # Function in drone.py
    cap = cv.VideoCapture(0)

    mediapipe_utils.initialise_hands()
    mediapipe_utils.initialise_pose()
    
    while True:
        key = cv.waitKey(1)
        # Read image from drone camera
        # image = drone.get_camera_image(drone_instance)  # Function in drone.py
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        ret, image = cap.read()
        
        # Extract pose and hands using MediaPipe
        hands, right_hand_roi = mediapipe_utils.extract_pose_and_hands(image)
        cv.imshow('ThirdPerson', image)
        if hands:
            cv.imshow('Right Hand', right_hand_roi)
        
        # Recognize gesture using neural network
        gesture = gesture_recognition.recognize_gesture(hands)  # Function in gesture_recognition.py
        
        # Map recognized gesture to instructions
        # instruction = gesture_instructions.map_to_instruction(gesture)  # Function in gesture_instructions.py
        
        # Execute instruction
        # drone.execute_instruction(instruction)  # Function in drone.py
        
        # Break the loop based on some condition (optional)
        if key == ord('q'):
            # drone.terminate_drone(drone_instance)
            webcam_functions.terminate_drone()
            mediapipe_utils.terminate_hands()
            mediapipe_utils.terminate_pose()
            break
        elif key == ord(' '):
            # drone.initialise_drone(drone_instance)
            pass

if __name__ == "__main__":
    main()