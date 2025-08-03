import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score
import numpy as np
import os
from keras.layers import Dropout
import time
start_time = time.time()

res = []
for i in range(1):

    # đọc dữ liệu
    path_train = "data/training_data.csv"
    path_label = "data/label_data.csv"
    df_feature = pd.read_csv(path_train, encoding='latin1')
    df_label = pd.read_csv(path_label, encoding='latin1')
    # df_feature = pd.read_csv(path_train, header=None, encoding='latin1')
    # df_label = pd.read_csv(path_label, header=None, encoding='latin1')
    label = df_label.values

    df_feature["group"] = df_feature.index // 60
    datasets = {g:v for g,v in df_feature.groupby("group")}

    X = np.stack([datasets[i].drop('group', axis=1).values for i in datasets.keys()])

    label = label.ravel()

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(label)

    # Chia dữ liệu thành tập huấn luyện (70%) và tập tạm thời (30%)
    x_temp, x_test, y_temp, y_test = train_test_split(X, Y, test_size=0.5)

    # Chia tập tạm thời thành tập kiểm định và tập kiểm tra, mỗi tập chiếm 50% tập tạm thời (tức là 15% dữ liệu gốc)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.5)

    # Chuyển đổi nhãn thành dạng one-hot

    num_word = len(label_encoder.classes_)

    #tạo callback
    tensorboard = TensorBoard(log_dir=os.path.join("logs"))

    # tạo mô hình RNN
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True, activation='relu',input_shape=(X.shape[1], X.shape[2])))
    model.add(SimpleRNN(64, return_sequences=True, activation='relu'))
    model.add(SimpleRNN(32, return_sequences=False, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_word, activation='softmax'))
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.fit(x_train, y_train, epochs=175, validation_data=(x_val, y_val), callbacks=[tensorboard])
    model.summary()
    model.save("model.h5")

    predictions = model.predict(x_test)

    predicted_classes = np.argmax(predictions, axis=1)
    print(len(y_test))
    res.append(accuracy_score(y_test, predicted_classes))
    # acc = accuracy_score(y_test, predicted_classes)

end_time = time.time()
elapsed_time = end_time - start_time
# print(elapsed_time)
print('độ chính xác trung bình là:', sum(res)/1)

