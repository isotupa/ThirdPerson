import csv
import copy
import itertools
import tensorflow as tf
import numpy as np

class GestureRecognizer(object):
    def __init__(self, model_path='model/keypoint_classifier.tflite', label_path='model/keypoint_classifier_label.csv'):
        self.keypoint_classifier, self.keypoint_classifier_labels = self.load_model(model_path=model_path, label_path=label_path)

    def recognize_gesture(self, results, debug_image):
        gesture = -1

        # if results is None or debug_image is None or results.multi_hand_landmarks is None:
        if results is None or debug_image is None or results is None or not hasattr(results, 'hand_landmarks'):
            return gesture, self.keypoint_classifier_labels

        # for hand_landmarks in results.multi_hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
            hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
            gesture = hand_sign_id

        return gesture, self.keypoint_classifier_labels

    def translate_gesture_id_to_name(self, gesture_id):
        if gesture_id == -1:
            return 'No gesture'
        return self.keypoint_classifier_labels[gesture_id]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        # for _, landmark in enumerate(landmarks.landmark):
        for landmark in landmarks:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def load_model(self, model_path, label_path):
        self.keypoint_classifier = KeyPointClassifier(model_path=model_path)

        with open(label_path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        return self.keypoint_classifier, self.keypoint_classifier_labels

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


class KeyPointClassifier(object):
    def __init__(self, model_path, num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
