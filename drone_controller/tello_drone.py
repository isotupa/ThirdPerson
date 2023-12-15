from djitellopy import Tello
import threading
import cv2 as cv
import time

from drone_controller.drone_interface import DroneController

class TelloDroneController(DroneController):
    def __init__(self):
        self.drone = None
        self.takeoff = False
    
    def get_drone(self):
        return self.drone

    def connect_to_drone(self):
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        return self.drone

    def get_camera_image(self):
        if self.drone:
            frame = self.drone.get_frame_read().frame
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            frame = cv.flip(frame, 1)
            return frame
        return None

    def initialise_drone(self):
        if self.drone:
            threading.Thread(target=self.drone.takeoff).start()
            self.takeoff = True
            time.sleep(1)
            threading.Thread(target=self.drone.move_up, args=(50)).start()

    def terminate_drone(self):
        if self.drone:
            self.drone.streamoff()
            self.drone.end()

    def execute_instruction(self, move):
        if self.drone:
            threading.Thread(target=self.drone.send_rc_control, args=move).start()

    def execute_roll(self):
        self.drone.flip_left()

    def get_battery(self):
        return self.drone.get_battery()

    def land(self):
        if self.takeoff:
            threading.Thread(target=self.drone.land).start()