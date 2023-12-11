from drone_controller import tello_drone
from drone_controller import webcam_drone
from drone_controller import mediapipe_utils
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
from mp_utils import pose_hands
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

    hand_pose_detection = pose_hands.HandPoseDetection()

    instructions = gesture_instructions.Instructions()
    buffer = gesture_buffer.GestureBuffer(buffer_len=10)

    gesture_recognizer = gesture_recognition.GestureRecognizer()

    # desired_fps = 15
    # interval = 1.0 / desired_fps

    
    while True:
        # start_time = time.time()

        image = drone.get_camera_image()
        battery = drone.get_battery()
        # print(battery)

        gesture = -1

        pose_result = hand_pose_detection.extract_pose(image)
        main_window_image = hand_pose_detection.draw_pose(image)
        right_hand_roi = hand_pose_detection.extract_right_hand_roi(image)
        hand_result = hand_pose_detection.extract_hands(right_hand_roi)
        hand_window_image = hand_pose_detection.draw_hands(right_hand_roi)

        gesture_id, labels = gesture_recognizer.recognize_gesture(hand_result, image)
        gesture_name = gesture_recognizer.translate_gesture_id_to_name(gesture_id)
        buffer.add_gesture(gesture_id)
        gesture = buffer.get_gesture()


        cv.putText(main_window_image, f'{instructions.get_follow_state()}', (330, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.imshow('ThirdPerson', main_window_image)
        if right_hand_roi is not None:
            cv.putText(hand_window_image, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
            cv.imshow('Right hand', hand_window_image)

        type_move, move = instructions.calculate_move(gesture, pose_result, image)
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
        
        # elapsed_time = time.time() - start_time
        # time_to_wait = max(0, interval - elapsed_time)
        # time.sleep(time_to_wait)

    drone.terminate_drone()
    hand_pose_detection.close()


if __name__ == "__main__":
    main()