import time
import math
import numpy as np

class Instructions():
    takeoff = False
    follow_behaviour = True
    moving = False
    speed = 10
    previous_move = (0,0,0,0)
    desired_distance = 300
    forward_backward_threshold = 50

    def __init__(self, speed=10):
        self.speed = speed

    def get_follow_state(self):
        return self.follow_behaviour

    def get_takeoff_state(self):
        return self.takeoff

    def calculate_move(self, gesture_id, pose, image):
        if gesture_id == None:
            return self.previous_move

        if self.follow_behaviour:
            if gesture_id == 1:
                self.follow_behaviour = not self.follow_behaviour
            return self.follow(pose, image)
        else:
            match gesture_id:
                case 0: # Forward
                    return (0, self.speed, 0, 0)
                case 1: # Stop
                    self.follow_behaviour = not self.follow_behaviour
                    return (0,0,0,0)
                case 2: # Up
                    return (0,0,self.speed, 0)
                case 3: # Land
                    self.follow_behaviour = not self.follow_behaviour
                # case 3: # Land
                #     return "land", 0 # TODO
                case 4: # Down
                    return (0,0,-self.speed, 0)
                case 5: # Back
                    return (0, -self.speed, 0, 0)
                case 6: # Left
                    return (self.speed, 0,0,0)
                case 7: # Right
                    return (-self.speed, 0,0,0)
                case 8: # Toggle follow
                    return self.follow(pose, image)
                case 9: # Semicircle
                    return self.semicircle()
                case 10: # change follow
                    return self.find_next_person()
                case _:
                    return (0,0,0,0)
    
    
    def calculate_velocity(self, x, width):
        center = width / 2
        max_velocity = 100  # Maximum velocity for movements
        distance = abs(center - x)
        # Scale velocity based on the distance from the center
        velocity = max_velocity * (distance / center)
        return int(velocity)

    def calculate_velocity_z(self, p1, p2):
        distance = math.sqrt(p1*p1 + p2*p2)
        threshold = self.forward_backward_threshold
        desired_distance = self.desired_distance
        max_speed = 100
        max_distance = 600
        if distance < desired_distance - threshold:
            return -(max_speed - ((100 / desired_distance-threshold) * distance))
        elif distance > desired_distance + threshold:
            return ((max_speed / (max_distance - desired_distance+threshold)) * (distance - max_distance))
        else:
            return 0


    def follow(self, pose, image):
        if pose is None:
            return (0,0,0,0)
        height, width, _ = image.shape
        keypoints = [(int(lm.x * width), int(lm.y * height)) for lm in pose.pose_landmarks.landmark]

        # Calculate speed based on the difference between current distance and desired distance
        nose_x, nose_y = keypoints[0] if keypoints else (0, 0)
        neck_x, neck_y = keypoints[12] if keypoints else (0,0)
        left_shoulder_x, left_shoulder_y = keypoints[11] if keypoints else (0, 0)
        right_shoulder_x, right_shoulder_y = keypoints[12] if keypoints else (0, 0)
        left_wrist_x, left_wrist_y = keypoints[7] if keypoints else (0, 0)
        right_wrist_x, right_wrist_y = keypoints[4] if keypoints else (0, 0)
        hip_x, _ = keypoints[24] if keypoints else (0, 0)
        
        # Define thresholds for movement detection
        center_x = width / 2
        center_y = height / 2
        threshold_x = 0.1 * center_x  # Adjust as needed
        threshold_y = 0.1 * center_y  # Adjust as needed
        
        # Calculate velocities based on distance from the center
        velocity_x = self.calculate_velocity(nose_x, width)
        velocity_y = self.calculate_velocity((left_shoulder_y + right_shoulder_y) / 2, height)
        # print(self.calculate_velocity_z(left_shoulder_x, right_shoulder_x))

        result = list((0,0,0,0))

        # Drone movement control based on pose keypoints
        if nose_x < center_x - threshold_x:
            result[0] = -velocity_x
            result[3] = -velocity_x
            # return (-velocity_x,0,0,-velocity_x)
            # drone.move_left(velocity_x)
        elif nose_x > center_x + threshold_x:
            result[0] = velocity_x
            result[3] = velocity_x
            # drone.move_right(velocity_x)
            # return (velocity_x,0,0,velocity_x)
        
        if nose_y < center_y - threshold_y:
            result[2] = velocity_y
            # return (0,0,velocity_y,0)
            # drone.move_up(velocity_y)
        elif nose_y > center_y + threshold_y:
            result[2] = -velocity_y
            # return (0,0,-velocity_y,0)
            # drone.move_down(velocity_y)
        # print(f'p1: {nose_x}, p2: {neck_x}')

        calc = self.calculate_velocity_z(nose_y, neck_y)
        result[1] = int(calc)

        return tuple(result)


    
    def semicircle(radius=0, speed=speed):
        circumference = 2 * math.pi * radius
        duration = circumference / (speed / 100)

        pitch = speed
        roll = speed / 2

        start_time = time.time()
        while time.time() - start_time < duration:
            return (-speed, 0, 0, speed)
        #     drone.send_rc_control(-speed,0,0,speed)
        #     time.sleep(0.05)

        # drone.send_rc_control(0,0,0,0)

    def find_next_person(self, pose, image):
        if pose is None:
            return (0,0,0,self.speed)
        return (0,0,0,0)