from drone_controller import tello_drone
from drone_controller import webcam_drone
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
from mp_utils import pose_hands
import cv2 as cv
import threading
import json
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

def overlay_text_on_rect(frame, text, rect_position, text_position, font_scale=1, color=(255, 255, 255), thickness=2):
    # Create a black rectangle as background for text
    rect_width = 300
    rect_height = 200
    rect_end_x = rect_position[0] + rect_width
    rect_end_y = rect_position[1] + rect_height
    cv.rectangle(frame, rect_position, (rect_end_x, rect_end_y), (0, 0, 0), -1)  # Draw filled black rectangle

    # Overlay text on the black rectangle
    cv.putText(frame, text, text_position, cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv.LINE_AA)


def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    if config['simulation']:
        drone = webcam_drone.WebcamSimulationController()
    else:
        drone = tello_drone.TelloDroneController()

    drone.connect_to_drone()

    hand_pose_detection = pose_hands.HandPoseDetection(
        hand_region_window_width=config['constants']['gui']['hand_window_width'],
        hand_region_window_height=config['constants']['gui']['hand_window_height'],
        pose_model_asset_path=config['model_paths']['pose_landmarker'],
        min_pose_detection_confidence=config['constants']['pose']['min_pose_detection_confidence'],
        min_pose_presence_confidence=config['constants']['pose']['min_pose_presence_confidence'],
        min_pose_tracking_confidence=config['constants']['pose']['min_tracking_confidence'],
        safe_zone=config['constants']['pose']['safe_zone'],
        hand_model_asset_path=config['model_paths']['hand_landmarker'],
        min_hand_presence_confidence=config['constants']['hands']['min_hand_presence_confidence'],
        min_hand_tracking_confidence=config['constants']['hands']['min_tracking_confidence'],
    )

    instructions = gesture_instructions.Instructions(
        following=config['initial_options']['following'],
        speed=config['constants']['speed']
    )
    buffer = gesture_buffer.GestureBuffer(buffer_len=config['constants']['buffer_length'])

    gesture_recognizer = gesture_recognition.GestureRecognizer(
        model_path=config['model_paths']['gesture_recogniser'],
        label_path=config['model_paths']['keypoint_classifier_labels']
    )

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

        type_move, move = instructions.calculate_move(gesture, pose_result, image)
        print(move)

        overlay_text_on_rect(main_window_image, f'{instructions.get_follow_state()}', (20, 20), (30, 60))
        overlay_text_on_rect(main_window_image, f'{move}', (20, 80), (30, 120))
        overlay_text_on_rect(main_window_image, f'{battery}', (20, 140), (30, 180))
        overlay_text_on_rect(main_window_image, f'{gesture_name}', (20, 200), (30, 240))

        # cv.putText(main_window_image, f'{instructions.get_follow_state()}', (330, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv.imshow('ThirdPerson', main_window_image)
        # if right_hand_roi is not None:
            # cv.putText(hand_window_image, f'Gesture: {gesture_name}', (0, 290), cv.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2)
        cv.imshow('Right hand', hand_window_image)

        
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