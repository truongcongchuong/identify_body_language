import cv2
import os
import sys
import numpy as np
import library as lb
import mediapipe as mp


#num of images in each action
frame_per_act = 60

#set a name for an added action
name_action_eng = input('nhập hành động mà bạn muốn thêm (không dấu): ')
name_action_vie = input('nhập hành động mà bạn muốn thêm (có dấu): ')

#num of actions that are looped 
num_action = int(input('nhập số lần lặp lại hành động đó(tùy vào độ khó của hành động): '))



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()


cv2.namedWindow('AddData', cv2.WINDOW_NORMAL)
cv2.waitKey(2000)

folder_path = f'image_data/{name_action_eng}'
num = 0

for action in range(num_action):
    for frame_num in range(frame_per_act):

        num += 1
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        os.makedirs(folder_path, exist_ok=True)

        cv2.imwrite(f'image_data/{name_action_eng}/{num}.jpg', frame)

        print(num)
        if frame_num == 0:
            cv2.putText(frame, 'CHUAN BI GHI HINH', (120,200),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow('AddData', frame)
            cv2.waitKey(2000)
        else:
            cv2.putText(frame, f'da ghi {action}/{num_action} ban ghi', (10,30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (230, 0, 0), 2, cv2.LINE_4)
            cv2.imshow('AddData', frame)
        cv2.imshow('AddData', frame)
        #extract keypoints
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


#################################################################################################################


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
detect_hand_pose = lb.detect_hand_pose
draw_landmarks = lb.draw_landmarks
extract_keypoints = lb.extract_keypoints
write_data = lb.write_data


file_list = os.listdir(folder_path)
total_frames = len(file_list)

with mp_hands.Hands() as hand:
    with mp_pose.Pose() as pose:
        new_action = []
        for idx, file_name in enumerate(file_list):
            file_path = os.path.join(folder_path, file_name)

            #visalize the progress
            if idx%10 == 0:
                
                # Tính phần trăm tiến trình
                progress = (idx / total_frames) * 100

                # Tính số lượng ký tự để in ra
                num_bars = int((idx / total_frames) * 50)

                # Tạo ra chuỗi biểu diễn tiến trình
                progress_str = '=' * (num_bars - 1) + '>' + ' ' * (50 - num_bars)

                # In ra biểu tượng load tiến trình
                sys.stdout.write("\rTiến trình: [{0}] {1:.1f}%".format(progress_str, progress))
                sys.stdout.flush()


            frame = cv2.imread(file_path)

            image, detected_hand, detected_pose = detect_hand_pose(frame, hand, pose)

            # extract keypoints
            if not (detected_hand.multi_hand_landmarks and detected_pose.pose_landmarks):
                key_points = np.zeros((201))
            else:
                key_points = extract_keypoints(detected_hand, detected_pose)

            new_action.append(key_points)


        # save data
        if (write_data('training_data.csv', 'label_data.csv', new_action, name_action_vie)):
            print("\n dữ liệu đã được lưu.")
        else:
            print("\n dữ liệu không được lưu.")

