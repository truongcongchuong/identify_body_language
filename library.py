import mediapipe as mp
import cv2
import numpy as np
import csv
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def detect_hand_pose(image, hand, pose):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    detected_hand = hand.process(image)                 # Make prediction
    detected_pose = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, detected_hand, detected_pose

def draw_landmarks(image, landmarks_hand, landmarks_pose):
    # Draw hand landmarks
    mp_drawing.draw_landmarks(image, landmarks_hand.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    if len(landmarks_hand.multi_hand_landmarks) == 2:
        mp_drawing.draw_landmarks(image, landmarks_hand.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)

    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, landmarks_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    

def write_data(feature_file_name, label_file_name, feature_data, label_data):
    try:
        with open("data/"+feature_file_name, mode='a', newline='') as feature:
            writer = csv.writer(feature)

            # write feature data into feature file
            writer.writerows(feature_data)

        with open("data/"+label_file_name, mode='a', newline='', encoding='utf-8') as label:
            writer = csv.writer(label)

            # write label data into label file
            for _ in range(len(feature_data)//60):
                writer.writerow([label_data])
        return True, None
    except Exception as e:
        return False, e
    
def extract_keypoints(detected_hand, detected_pose):
    landmarks_extract_pose = [pose for pose in range(25)]
    landmarks_pose = np.array([[res.x, res.y, res.z] for i, res in enumerate(detected_pose.pose_landmarks.landmark) if i in landmarks_extract_pose]).flatten()
    if len(detected_hand.multi_hand_landmarks) == 1:
        hand_left = (np.array([[res.x, res.y, res.z] for res in detected_hand.multi_hand_landmarks[0].landmark]).flatten() 
                        if detected_hand.multi_hand_landmarks[0] and detected_hand.multi_handedness[0].classification[0].label == "Left" else np.zeros((63)))

        hand_right = (np.array([[res.x, res.y, res.z] for res in detected_hand.multi_hand_landmarks[0].landmark]).flatten() 
                        if detected_hand.multi_hand_landmarks[0] and detected_hand.multi_handedness[0].classification[0].label == "Right" else np.zeros((63)))

    else:
        if detected_hand.multi_handedness[0].classification[0].label == "Left":
            hand_left = (np.array([[res.x, res.y, res.z] for res in detected_hand.multi_hand_landmarks[0].landmark]).flatten() 
                            if detected_hand.multi_handedness[0].classification[0].label == "Left" else np.zeros((63)))

            hand_right = (np.array([[res.x, res.y, res.z] for res in detected_hand.multi_hand_landmarks[1].landmark]).flatten() 
                            if detected_hand.multi_handedness[1].classification[0].label == "Right" else np.zeros((63)))
        else:
            hand_right = (np.array([[res.x, res.y, res.z] for res in detected_hand.multi_hand_landmarks[0].landmark]).flatten() 
                            if detected_hand.multi_handedness[0].classification[0].label == "Right" else np.zeros((63)))

            hand_left = (np.array([[res.x, res.y, res.z] for res in detected_hand.multi_hand_landmarks[1].landmark]).flatten() 
                            if detected_hand.multi_handedness[1].classification[0].label == "Left" else np.zeros((63)))
    
    final_hand_left = np.round(hand_left, 6)
    final_hand_right = np.round(hand_right, 6)
    return np.concatenate([final_hand_left, final_hand_right, landmarks_pose])
