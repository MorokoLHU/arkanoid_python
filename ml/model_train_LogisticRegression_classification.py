import pickle
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

# 讀取 log 檔資料夾
path = os.path.join(os.path.dirname(__file__), "..", "log")
all_files = os.listdir(path)
data_set = []

for file in all_files:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))

Ball_x, Ball_y, Vector_x, Vector_y, Direction, Platform, Command, Relative_pos = [], [], [], [], [], [], [], []

for data in data_set:
    scene_info = data["scene_info"]
    commands = data["command"]
    for i in range(2, len(scene_info) - 2):
        try:
            bx = scene_info[i]["ball"][0]
            by = scene_info[i]["ball"][1]
            px = scene_info[i]["platform"][0]
            vx = scene_info[i + 1]["ball"][0] - scene_info[i]["ball"][0]
            vy = scene_info[i + 1]["ball"][1] - scene_info[i]["ball"][1]

            # 方向分類
            if vx > 0 and vy > 0:
                direction = 0  # 右下
            elif vx > 0 and vy <= 0:
                direction = 1  # 右上
            elif vx <= 0 and vy > 0:
                direction = 2  # 左下
            else:
                direction = 3  # 左上

            # 指令標籤處理
            cmd = commands[i]
            if cmd in ["", "NONE"]:
                label = 0
            elif cmd == "MOVE_LEFT":
                label = -1
            elif cmd == "MOVE_RIGHT":
                label = 1
            else:
                raise ValueError("未知指令")

            # 收集資料
            Ball_x.append(bx)
            Ball_y.append(by)
            Vector_x.append(vx)
            Vector_y.append(vy)
            Direction.append(direction)
            Platform.append(px)
            Relative_pos.append(bx - px)
            Command.append(label)

        except:
            # 資料不完整，略過
            continue

# 組合特徵與標籤
X = np.array([Ball_x, Ball_y, Vector_x, Vector_y, Direction, Relative_pos]).T
Y = np.array(Command)

# 切分資料集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 從 PHP 接參數
php_C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

# 建立邏輯回歸模型
model = LogisticRegression(C=php_C, class_weight="balanced", max_iter=1000)
model.fit(x_train, y_train)

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

with open(os.path.join(save_path, "LogisticRegression_CL_acc={:.2f}.pickle".format(accuracy)), 'wb') as f:
    pickle.dump(model, f)
