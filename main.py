from drone_controller import tello_drone
from drone_controller import webcam_drone
from neural_network import gesture_recognition
from instructions import gesture_instructions
from instructions import gesture_buffer
from mp_utils import pose_hands
from gui import gui
import threading
import json


def control(key, tello):
    if key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('k'):
        tello.rotate_clockwise(30)
    elif key == ord('l'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    if config['simulation']:
        drone = webcam_drone.WebcamSimulationController(config['constants']['webcam_number'])
    else:
        drone = tello_drone.TelloDroneController()

    tp_gui = gui.ThirdPersonGUI(config['constants']['gui']['hand_window_height'], config['constants']['gui']['hand_window_width'])

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
    
    while True:

        image = drone.get_camera_image()
        battery = drone.get_battery()

        pose_result = hand_pose_detection.extract_pose(image)
        right_hand_roi = hand_pose_detection.extract_right_hand_roi(image)
        main_window_image = hand_pose_detection.draw_pose(image)
        hand_result = hand_pose_detection.extract_hands(right_hand_roi)
        hand_window_image = hand_pose_detection.draw_hands(right_hand_roi)

        gesture_id, _ = gesture_recognizer.recognize_gesture(hand_result, right_hand_roi)
        gesture_name = gesture_recognizer.translate_gesture_id_to_name(gesture_id)
        buffer.add_gesture(gesture_id)
        gesture = buffer.get_gesture()

        type_move, move = instructions.calculate_move(gesture, pose_result, image)

        tp_gui.update_camera_window(main_window_image)
        tp_gui.update_hand_window(hand_window_image)
        tp_gui.update_info_window(instructions.get_follow_state(),
                                  move,
                                  battery,
                                  gesture_name)
        tp_gui.show_window()
        key = tp_gui.getKey()

        if type_move == 'tuple':
            drone.execute_instruction(move)
        elif type_move == 'land':
            drone.land()
            tp_gui.landing()
            tp_gui.show_window()
            break
        elif type_move == 'roll':
            drone.execute_roll()
        
        if key == ord('l'):
            drone.land()
        elif key == ord(' '):
            drone.initialise_drone()
        elif key == ord('f'):
            instructions.follow_behaviour = not instructions.follow_behaviour
        elif key == ord('p'):
            drone.execute_roll()
        elif key == ord('q'):
            break
        else:
            threading.Thread(target=control, args=(key,drone.get_drone())).start()

    drone.terminate_drone()
    hand_pose_detection.close()
    tp_gui.close()


if __name__ == "__main__":
    main()