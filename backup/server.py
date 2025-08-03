from flask import Flask, request, render_template
import cv2
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model
import pandas as pd
import library as lb
import mediapipe as mp
import time

try:
    model = load_model("model.h5")
    print("Model loaded")
except Exception as e:
    print(str(e))


label = pd.read_csv("data/label_data.csv", header=None, encoding="utf-8").values
label = label.ravel()
label_encoder = LabelEncoder()
actions = label_encoder.fit_transform(label)
frame_per_act = 40
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("web_ditector.html")

@app.route('/frame', methods=["GET","POST"])
def HandleVideo():
    try: 
        centence = []
        sequence = []  
        video = request.files['video']
        filename = secure_filename(video.filename)
        print(filename)
        # file_path = os.path.join('C:/Users/ThinkPad T480/Videos/Captures', filename)
        # video.save(file_path)
        video.save(filename)
        cap = cv2.VideoCapture(f"./{filename}")
        
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f'Total frames: {total_frames}')
        # cap.set(cv2.CAP_PROP_POS_MSEC, 3000)  # Set thời gian chạy video là 3 giây
        # start_time = time.time()  # Lưu thời gian bắt đầu
        
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        
        
        with mp_hands.Hands() as hand:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                # start_time = round(time.perf_counter(),2)
                count = 0
                while cap.isOpened():
                    count += 1
                    print("số frame được xử lý là:",count)
                    ret, frame = cap.read()
                    # if time.time() - start_time > 10:  # Kiểm tra xem đã qua 3 giây chưa
                    #     break
                    if frame is None:
                        break
                    frame = cv2.flip(frame, 1)
                    image, detected_hand, detected_pose = lb.detect_hand_pose(frame, hand, pose)
                    if detected_hand.multi_hand_landmarks and detected_pose.pose_landmarks:
                        key_points = lb.extract_keypoints(detected_hand, detected_pose)
                        sequence.append(key_points)
                        
                        
                    if len(sequence) == 40:
                        prediction = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predicted_classes = np.argmax(prediction)
                        predicted_classes = np.array([predicted_classes])
                        predictions_result = str(label_encoder.inverse_transform(predicted_classes))
                        # time_record = round(time.perf_counter(),2) - start_time
                        # print('số khung hình trên giây:', frame_per_act/time_record)
                        if len(centence) == 0:
                            centence.append(predictions_result)
                        else:
                            if centence[-1] != predictions_result:
                                centence.append(predictions_result)
                        sequence = sequence[-40:]
                cap.release() 
                
                cv2.destroyAllWindows()
        os.remove(filename)  # Xóa file video sau khi xử lý
        res = ' '.join(centence)
        res = res.replace('[', '').replace(']', '').replace("'", "")
        print("FPS of the video is:", fps)
        # In tổng số frame
        return res
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)

































































