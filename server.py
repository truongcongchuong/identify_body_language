from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
import pandas as pd
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)  # Cho phép các yêu cầu CORS từ các domain khác, ví dụ như ứng dụng React Native

# Load model Keras
try:
    model = load_model("model.h5")
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", str(e))

# Load label encoder
label = pd.read_csv("data/label_data.csv", header=None, encoding="utf-8").values.ravel()
label_encoder = LabelEncoder()
actions = label_encoder.fit_transform(label)
frame_per_act = 60

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

ALLOWED_EXTENSIONS = {'mp4', 'avi'}  # Các định dạng video cho phép

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=["POST"])
def handle_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)
            
            sequence = []
            centence = []

            cap = cv2.VideoCapture(filename)
            with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.flip(frame, 1)
                        image, detected_hand, detected_pose = lb.detect_hand_pose(frame, hands, pose)

                        if detected_hand.multi_hand_landmarks and detected_pose.pose_landmarks:
                            key_points = lb.extract_keypoints(detected_hand, detected_pose)
                            sequence.append(key_points)
                        else:
                            key_points = np.zeros((201))

                        if len(sequence) == 60:
                            prediction = model.predict(np.expand_dims(sequence, axis=0))[0]
                            if np.max(prediction) < 0.75:
                                sequence = sequence[-55:]
                            else:
                                predicted_class = np.argmax(prediction)
                                predicted_class = np.array([predicted_class])
                                prediction_result = label_encoder.inverse_transform(predicted_class)
                                
                                if len(centence) == 0:
                                    centence.append(prediction_result)
                                else:
                                    if centence[-1] != prediction_result:
                                        centence.append(prediction_result)
                                    
                                sequence = sequence[-20:]

            cap.release()
            cv2.destroyAllWindows()
            os.remove(filename)

            result = ' '.join(centence)
            result = result.replace('[', '').replace(']', '').replace("'", "")

            return jsonify({'result': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8083)
