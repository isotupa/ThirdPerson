# from drone_interface import DroneController
import cv2 as cv

from drone_controller.drone_interface import DroneController

class WebcamSimulationController(DroneController):
    def __init__(self, webcam_number=0):
        self.drone = None
        self.takeoff = False
        self.webcam_number = webcam_number

    def connect_to_drone(self):
        cap = cv.VideoCapture(self.webcam_number)
        self.drone = cap
        return self.drone

    def get_camera_image(self):
        ret, image = self.drone.read()
        image = cv.flip(image, 1)
        return image
    
    def get_drone(self):
        return None

    def initialise_drone(self):
        pass

    def terminate_drone(self):
        self.drone.release()

    def execute_instruction(self, move):
        pass

    def execute_roll(self):
        pass

    def get_battery(self):
        return 100

    def land(self):
        pass
