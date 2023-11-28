from abc import ABC, abstractmethod

class DroneController(ABC):
    @abstractmethod
    def connect_to_drone(self):
        pass

    @abstractmethod
    def get_camera_image(self):
        pass

    @abstractmethod
    def initialise_drone(self):
        pass

    @abstractmethod
    def terminate_drone(self):
        pass

    @abstractmethod
    def execute_instruction(self, move):
        pass
