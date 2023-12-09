import time
import math
import numpy as np
import cv2 as cv

class Instructions():
    takeoff = False
    follow_behaviour = False
    moving = False
    previous_move = (0,0,0,0)
    desired_distance = 200
    forward_backward_threshold = 25


    def __init__(self, speed=40, width=1000, height=500):
        self.speed = speed
        self.width = width
        self.height = height
        self.center_x = width / 2
        self.center_y = height / 2

    def get_follow_state(self):
        return self.follow_behaviour

    def get_takeoff_state(self):
        return self.takeoff

    def calculate_move(self, gesture_id, pose, image):
        # if gesture_id == None:
        #     return 'tuple', (0,0,0,0)

        # if self.follow_behaviour:
        #     match gesture_id:
        #         case 0: # Forward
        #             self.desired_distance += 10
        #         case 1: # Stop
        #             self.follow_behaviour = not self.follow_behaviour
        #         case 2: # Up
        #             self.center_y += 10
        #         case 3: # Land
        #             return 'land', None
        #         case 4: # Down
        #             self.center_y -= 10
        #         case 5: # Back
        #             self.desired_distance -= 10
        #         case 6: # Left
        #             self.center_x += 10
        #         case 7: # Right
        #             self.center_x -= 10
        #         case 8: # Toggle follow
        #             self.follow_behaviour = not self.follow_behaviour
        #         case 9: # Semicircle
        #             self.follow_behaviour = not self.follow_behaviour
        #             return 'tuple', self.semicircle()
        #         case 10: # change follow
        #             self.follow_behaviour = not self.follow_behaviour
        #             return 'tuple', self.find_next_person()
        #         case 11: # roll
        #             return 'roll', None
        #     return 'tuple', self.follow(pose, image)

        move = self.previous_move
        match gesture_id:
            case 0: # Forward
                move = (0, self.speed, 0, 0)
            case 1: # Stop
                move = (0,0,0,0)
            case 2: # Up
                move = (0,0,self.speed, 0)
            case 3: # Land
                return 'land', None
            case 4: # Down
                move = (0,0,-self.speed, 0)
            case 5: # Back
                move = (0, -self.speed, 0, 0)
            case 6: # Left
                move = (self.speed, 0,0,0)
            case 7: # Right
                move = (-self.speed, 0,0,0)
            case 8: # Toggle follow
                self.follow_behaviour = not self.follow_behaviour
                move = self.follow(pose, image)
            case 9: # Semicircle
                move = (0,0,0,0)
            case 10: # change follow
                return 'tuple', (0,0,0,0)
                return 'tuple', self.find_next_person()
            case 11: # roll
                # return 'tuple', (0,0,0,0)
                return 'roll', None
        self.previous_move = move
        return 'tuple', move
    
    
    def calculate_velocity(self, x, width):
        center = width / 2
        max_velocity = 60  # Maximum velocity for movements
        distance = abs(center - x)
        # Scale velocity based on the distance from the center
        velocity = max_velocity * (distance / center)
        return int(velocity)


    def calculate_velocity_z_2(self, nose, left_neck, right_neck):
        left_distance = math.sqrt((nose[0]-left_neck[0])**2 + (nose[1]-left_neck[1])**2)
        right_distance = math.sqrt((nose[0]-right_neck[0])**2 + (nose[1]-right_neck[1])**2)
        mean = (left_distance + right_distance) / 2
        return mean


    def calculate_velocity_z_3(self, nose, left_neck, right_neck):
        midpoint = (left_neck[0], (left_neck[1] + right_neck[1])/2)
        distance = math.sqrt((nose[0]-midpoint[0])**2 + (nose[1]-midpoint[1])**2)
        return distance


    def calculate_velocity_z(self, nose, left, right):
        # distance = abs(nose_y - neck_y)
        distance = int(self.calculate_velocity_z_2(nose, left, right))
        threshold = self.forward_backward_threshold
        max_speed = 60
        max_distance = 720

        # Adjust these values as necessary
        desired_distance = self.desired_distance

        if distance < desired_distance - threshold:
            # Simplified calculation for moving backward
            velocity = max(-max_speed, -max_speed * ((distance - desired_distance) / (max_distance - desired_distance - threshold)))
            velocity = -velocity
    
        elif distance > desired_distance + threshold:
            # Simplified calculation for moving forward
            velocity = min(max_speed, max_speed * ((distance - desired_distance) / (max_distance - desired_distance - threshold)))
        else:
            velocity = 0

        return -velocity

    def follow(self, pose, image):
        if pose is None or pose.pose_landmarks is None:
            return (0,0,0,0)
        height, width, _ = image.shape
        keypoints = [(int(lm.x * width), int(lm.y * height)) for lm in pose.pose_landmarks.landmark]

        # Calculate speed based on the difference between current distance and desired distance
        nose_x, nose_y = keypoints[0] if keypoints else (0, 0)
        neck_x, neck_y = keypoints[12] if keypoints else (0,0)
        neck1_x, neck1_y = keypoints[11] if keypoints else (0,0)
        left_shoulder_x, left_shoulder_y = keypoints[11] if keypoints else (0, 0)
        right_shoulder_x, right_shoulder_y = keypoints[12] if keypoints else (0, 0)
        left_wrist_x, left_wrist_y = keypoints[7] if keypoints else (0, 0)
        right_wrist_x, right_wrist_y = keypoints[4] if keypoints else (0, 0)
        hip_x, _ = keypoints[24] if keypoints else (0, 0)
        
        # Define thresholds for movement detection
        # center_x = width / 2
        # center_y = height / 2
        center_x = self.center_x
        center_y = self.center_y
        threshold_x = 0.1 * center_x  # Adjust as needed
        threshold_y = 0.1 * center_y  # Adjust as needed
        
        # Calculate velocities based on distance from the center
        velocity_x = self.calculate_velocity(nose_x, width)
        velocity_y = self.calculate_velocity((left_shoulder_y + right_shoulder_y) / 2, height)
        # print(self.calculate_velocity_z(left_shoulder_x, right_shoulder_x))

        result = list((0,0,0,0))

        # Drone movement control based on pose keypoints
        if nose_x < center_x - threshold_x:
            result[0] = -int(velocity_x/2)
            result[3] = -velocity_x
            # return (-velocity_x,0,0,-velocity_x)
            # drone.move_left(velocity_x)
        elif nose_x > center_x + threshold_x:
            result[0] = int(velocity_x/2)
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
        average = int(self.calculate_velocity_z_2(keypoints[0], keypoints[12], keypoints[11]))
        tri_height =int(self.calculate_velocity_z_3(keypoints[0], keypoints[12], keypoints[11])) 
        calc = int(self.calculate_velocity_z(keypoints[0], keypoints[12], keypoints[11]))
        # print(f'mediapipe depth: {pose.pose_landmarks.landmark[0].z}')
        # print(f'mediapipe world: {pose.pose_world_landmarks.landmark[0].z}')
        print(f'triangle average:{average}')
        # print(f'triangle height:{tri_height}')
        # print(f'classic: {calc}')
        result[1] = int(calc)
        
        cv.line(image, (neck_x, neck_y), (nose_x, nose_y), (255,0,0),2)
        cv.line(image, (neck1_x, neck1_y), (nose_x, nose_y), (255,0,0),2)


        return tuple(result)


    
    def semicircle(radius=0, speed=30):
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