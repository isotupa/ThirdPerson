from drone_controller import tello_drone
from drone_controller import webcam_drone
from drone_controller import mediapipe_utils
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
import cv2 as cv
import threading
# import time


def control(key, tello):
    if key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)

def main():
    drone = webcam_drone.WebcamSimulationController()
    # drone = tello_drone.TelloDroneController()
    drone.connect_to_drone()

    mediapipe_utils.initialise_hands()
    mediapipe_utils.initialise_pose()

    instructions = gesture_instructions.Instructions()
    buffer = gesture_buffer.GestureBuffer(buffer_len=5)

    gesture_recognizer = gesture_recognition.GestureRecognizer()

    # desired_fps = 15
    # interval = 1.0 / desired_fps

    
    while True:
        # start_time = time.time()

        image = drone.get_camera_image()
        battery = drone.get_battery()
        print(battery)

        gesture = -1

        pose_results = mediapipe_utils.extract_pose(image)
        right_hand_roi = mediapipe_utils.extract_hand_region(image, pose_results)
        hands_results = mediapipe_utils.extract_hands(right_hand_roi)
        # right_hand = mediapipe_utils.draw_hands(right_hand_roi, hands_results)
        debug_image = mediapipe_utils.draw_landmarks_on_image(right_hand_roi, hands_results)
        mediapipe_utils.draw_pose(image, pose_results)
        gesture_id, labels = gesture_recognizer.recognize_gesture(hands_results, image)
        gesture_name = gesture_recognizer.translate_gesture_id_to_name(gesture_id)
        buffer.add_gesture(gesture_id)
        gesture = buffer.get_gesture()


        cv.putText(image, f'{instructions.get_follow_state()}', (330, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.imshow('ThirdPerson', image)
        if right_hand_roi is not None:
            # cv.putText(right_hand_roi, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
            # cv.putText(debug_image, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
            cv.imshow('Right hand', debug_image)

        type_move, move = instructions.calculate_move(gesture, pose_results, image)
        print(move)
        
        # if not instructions.get_takeoff_state() and type_move == 'tuple':
        if type_move == 'tuple':
            drone.execute_instruction(move)
        elif type_move == 'land':
            break
        elif type_move == 'roll':
            drone.execute_roll()
        
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            drone.initialise_drone()
        else:
            threading.Thread(target=control, args=(key,drone.get_drone())).start()
            # control(key, drone.get_drone())
        
        # elapsed_time = time.time() - start_time
        # time_to_wait = max(0, interval - elapsed_time)
        # time.sleep(time_to_wait)

    drone.terminate_drone()
    mediapipe_utils.terminate_hands()
    mediapipe_utils.terminate_pose()


if __name__ == "__main__":
    main()