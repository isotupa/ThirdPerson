# drone_controller/drone.py
from djitellopy import Tello

global takeoff
takeoff = False

def connect_to_drone():
    # Create an instance of the Tello class
    drone = Tello()

    # Connect to the drone
    drone.connect()

    # Enable video streaming
    drone.streamon()

    return drone

def get_camera_image(drone):
    # Get the current frame from the drone's video stream
    frame = drone.get_frame_read().frame

    return frame

def initialise_drone(drone: Tello):
    drone.takeoff()
    global takeoff
    takeoff = True


def terminate_drone(drone: Tello):
    if takeoff:
        drone.land()
    drone.streamoff()
    drone.end()