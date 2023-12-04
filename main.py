from drone_controller import tello_drone
from drone_controller import webcam_drone
from drone_controller import mediapipe_utils
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
import cv2 as cv


def main():
    drone = webcam_drone.WebcamSimulationController()
    # drone = tello_drone.TelloDroneController()
    drone.connect_to_drone()

    mediapipe_utils.initialise_hands()
    mediapipe_utils.initialise_pose()

    instructions = gesture_instructions.Instructions()
    buffer = gesture_buffer.GestureBuffer()
    
    while True:
        image = drone.get_camera_image()
        battery = drone.get_battery()
        # print(battery)

        gesture = -1
        
        pose, hands, right_hand_roi = mediapipe_utils.extract_pose_and_hands(image)
        if hands:
            gesture, labels = gesture_recognition.recognize_gesture(hands, right_hand_roi)
            gesture_name = gesture_recognition.translate_gesture_id_to_name(gesture, labels)
            buffer.add_gesture(gesture)
            gesture = buffer.get_gesture()
            
            cv.putText(right_hand_roi, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)

            cv.imshow('Right Hand', right_hand_roi)
        
        type_move, move = instructions.calculate_move(gesture, pose, image)
        print(move)
        
        # if not instructions.get_takeoff_state() and type_move == 'tuple':
        if type_move == 'tuple':
            drone.execute_instruction(move)
        elif type_move == 'land':
            break
        elif type_move == 'roll':
            drone.execute_roll()
        cv.putText(image, f'{instructions.get_follow_state()}', (330, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.imshow('ThirdPerson', image)
        
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            drone.initialise_drone()

    drone.terminate_drone()
    mediapipe_utils.terminate_hands()
    mediapipe_utils.terminate_pose()
if __name__ == "__main__":
    main()