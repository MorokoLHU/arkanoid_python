import pickle
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression  
path = os.path.join(os.path.dirname(__file__), "..", "log")
allFile = os.listdir(path)
data_set = []

for file in allFile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))

Ball_x = []
Ball_y = []
Vector_x = []
Vector_y = []
Direction = []
Platform = []
Command = []

# 解析數據
for data in data_set:
    for i, sceneInfo in enumerate(data["scene_info"][2:-2]):
        Ball_x.append(sceneInfo['ball'][0])
        Ball_y.append(sceneInfo['ball'][1])
        Platform.append(sceneInfo['platform'][0])
        Vector_x.append(data['scene_info'][i+1]["ball"][0] - data['scene_info'][i]["ball"][0])
        Vector_y.append(data['scene_info'][i+1]["ball"][1] - data['scene_info'][i]["ball"][1])

        if Vector_x[-1] > 0:
            if Vector_y[-1] > 0: Direction.append(0)  # 右下
            else: Direction.append(1)  # 右上
        else:
            if Vector_y[-1] > 0: Direction.append(2)  # 左下
            else: Direction.append(3)  # 左上

    for command in data['command'][2:-2]:
        if command == "NONE" or command == "":
            Command.append(0)  # 不動
        elif command == "MOVE_LEFT":
            Command.append(-1)  # 左移
        elif command == "MOVE_RIGHT":
            Command.append(1)  # 右移
        else:
            Ball_x.pop()
            Ball_y.pop()
            Platform.pop()
            Vector_x.pop()
            Vector_y.pop()
            Direction.pop()

Command = np.array(Command)
Command = Command.reshape(len(Command), 1)

# 特徵 X
X = np.array([Ball_x, Ball_y, Vector_x, Vector_y, Direction]).T

# 目標 Y（指令）
Y = Command[:, 0]

# 資料切分（8:2 訓練 & 測試）
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#### **線性分類（邏輯回歸）** ####
php_C=int(sys.argv[1])          #使其接收PHP的參數


model = LogisticRegression(C=php_C)  
model.fit(x_train, y_train)  # 訓練模型

# 預測與評估
y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='weighted')

print("Logistic Regression Accuracy = %.2f" % accuracy)
print("Logistic Regression F1 Score = %.2f" % f1)

# 儲存模型
save_path = os.path.join(os.path.dirname(__file__), 'save')
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, "LogisticRegression_acc={:.2f}_data={}.pickle".format(accuracy, len(X))), 'wb') as f:
    pickle.dump(model, f)
