# from drone_interface import DroneController
import cv2 as cv

from drone_controller.drone_interface import DroneController

class WebcamSimulationController(DroneController):
    def __init__(self):
        self.drone = None
        self.takeoff = False

    def connect_to_drone(self):
        cap = cv.VideoCapture(0)
        self.drone = cap
        return self.drone

    def get_camera_image(self):
        ret, image = self.drone.read()
        return image

    def initialise_drone(self):
        pass

    def terminate_drone(self):
        self.drone.release()

    def execute_instruction(self, move):
        pass
