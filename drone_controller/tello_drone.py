from djitellopy import Tello
import threading
import cv2 as cv

from drone_controller.drone_interface import DroneController

class TelloDroneController(DroneController):
    def __init__(self):
        self.drone = None
        self.takeoff = False

    def connect_to_drone(self):
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        return self.drone

    def get_camera_image(self):
        if self.drone:
            frame = self.drone.get_frame_read().frame
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            return frame
        return None

    def initialise_drone(self):
        if self.drone:
            self.drone.takeoff()
            self.takeoff = True
            self.drone.move_up(50)

    def terminate_drone(self):
        if self.drone:
            if self.takeoff:
                self.drone.land()
            self.drone.streamoff()
            self.drone.end()

    def execute_instruction(self, move):
        if self.drone:
            threading.Thread(target=self.drone.send_rc_control, args=move).start()

    def execute_roll(self):
        self.drone.flip_left()

    def get_battery(self):
        return self.drone.get_battery()


'''
class TelloDroneController(DroneController):
    def __init__(self):
        self.drone = None
        self.takeoff = False

    def connect_to_drone(self):
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        return self.drone

    def get_camera_image(self):
        if self.drone:
            frame = self.drone.get_frame_read().frame
            return frame
        return None

    def initialise_drone(self):
        if self.drone:
            self.drone.takeoff()
            self.takeoff = True

    def terminate_drone(self):
        if self.drone:
            if self.takeoff:
                self.drone.land()
            self.drone.streamoff()
            self.drone.end()

    def execute_instruction(self, move):
        if self.drone:
            threading.Thread(target=self.drone.send_rc_control, args=move).start()
'''