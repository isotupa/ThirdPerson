import cv2

def connect_to_drone():
    cap = cv2.VideoCapture(0)  # Access the default webcam (change the index if you have multiple)

    return cap

def get_camera_image(cap):
    # Get the current frame from the drone's video stream
    ret, frame = cap.read()

    return frame



def terminate_drone(cap):
    cap.release()