from drone_controller import tello_drone
from drone_controller import webcam_drone
from drone_controller import mediapipe_utils
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
import cv2 as cv


def main():
    drone = webcam_drone.WebcamSimulationController()
    drone.connect_to_drone()

    mediapipe_utils.initialise_hands()
    mediapipe_utils.initialise_pose()

    instructions = gesture_instructions.Instructions()
    buffer = gesture_buffer.GestureBuffer()
    
    while True:
        image = drone.get_camera_image()

        gesture = -1
        
        pose, hands, right_hand_roi = mediapipe_utils.extract_pose_and_hands(image)
        if hands:
            gesture, labels = gesture_recognition.recognize_gesture(hands, right_hand_roi)
            gesture_name = gesture_recognition.translate_gesture_id_to_name(gesture, labels)
            buffer.add_gesture(gesture)
            gesture = buffer.get_gesture()
            
            cv.putText(right_hand_roi, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (150,0,0), 2)

            cv.imshow('Right Hand', right_hand_roi)
        
        move = instructions.calculate_move(gesture, pose, image)
        print(move)
        
        if not instructions.get_takeoff_state():
            drone.execute_instruction(move)
        cv.putText(image, f'{instructions.get_follow_state()}', (330, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.imshow('ThirdPerson', image)
        
        key = cv.waitKey(1)
        if key == ord('q'):
            drone.terminate_drone()
            mediapipe_utils.terminate_hands()
            mediapipe_utils.terminate_pose()
            break
        elif key == ord(' '):
            drone.initialise_drone()

if __name__ == "__main__":
    main()