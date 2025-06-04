# import pickle
# import numpy as np
# import os
# import sys
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.ensemble import RandomForestClassifier

# # 讀取數據
# path = os.path.join(os.path.dirname(__file__), "..", "log")
# allfile = os.listdir(path)
# data_set = []

# for file in allfile[:]:
#     with open(os.path.join(path, file), "rb") as f:
#         data_set.append(pickle.load(f))

# Ball_x, Ball_y, Vector_x, Vector_y, Direction, Platform, command = [], [], [], [], [], [], []

# for data in data_set:
#     for i, sceneInfo in enumerate(data["scene_info"][2:-2]):
#         Ball_x.append(sceneInfo['ball'][0])
#         Ball_y.append(sceneInfo['ball'][1])
#         Platform.append(sceneInfo['platform'][0])
#         Vector_x.append(data['scene_info'][i + 1]["ball"][0] - data['scene_info'][i]["ball"][0])
#         Vector_y.append(data['scene_info'][i + 1]["ball"][1] - data['scene_info'][i]["ball"][1])
#         if Vector_x[-1] > 0:
#             Direction.append(0 if Vector_y[-1] > 0 else 1)
#         else:
#             Direction.append(2 if Vector_y[-1] > 0 else 3)
#     for Command in data['command'][2:-2]:
#         if Command == "NONE" or Command == "":
#             command.append(0)
#         elif Command == "MOVE_LEFT":
#             command.append(-1)
#         elif Command == "MOVE_RIGHT":
#             command.append(1)
#         else:
#             Ball_x = Ball_x[:-1]
#             Ball_y = Ball_y[:-1]
#             Platform = Platform[:-1]
#             Vector_x = Vector_x[:-1]
#             Vector_y = Vector_y[:-1]
#             Direction = Direction[:-1]

# command = np.array(command).reshape(len(command), 1)

# # 建立特徵與標籤
# X = np.array([0, 0, 0, 0, 0, 0])
# for i in range(len(Ball_x)):
    
#     X = np.vstack((X, [Ball_x[i], Ball_y[i], Vector_x[i], Vector_y[i], Direction[i],Platform[i]]))
# X = X[1::]
# Position_pred = []
# platform_position_y = 400
# ball_speed_y = 7
# platform_width = 200
# for i in range(len(Ball_x)):
#     pred = Ball_x[i] + ((platform_position_y - Ball_y[i]) // ball_speed_y) * Vector_x[i]
#     section = (pred // platform_width)
#     if (section % 2 == 0):
#         pred = abs(pred - platform_width * section)
#     else:
#         pred = platform_width - abs(pred - platform_width * section)

#     Position_pred.append(pred)
# Position_pred = np.array(Position_pred)
# Y = Position_pred
# print ("Data clear!\n")

# # 資料分割
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# # 隨機森林模型

# # 接收PHP參數
# php_max_depth = int(sys.argv[1]) 
# php_random_state =int(sys.argv[2]) 
# php_n_estimators = int(sys.argv[3])


# clf = RandomForestClassifier(max_depth=php_max_depth,
#                              random_state=php_random_state,
#                              n_estimators=php_n_estimators)
# clf.fit(x_train, y_train)
# RandomForestClassifier(...)
# print(clf.predict([[0, 0, 0, 0,0,0]]))
# # 驗證模型
# y_predict = clf.predict(x_val)
# Accuracy = accuracy_score(y_predict, y_val)
# F1Score = f1_score(y_val, y_predict, average='weighted')

# print("Random Forest Accuracy = {:.3f}".format(Accuracy))
# print("Random Forest F1 Score = {:.3f}".format(F1Score))

# # 測試模型
# y_test_predict = clf.predict(x_test)
# test_accuracy = accuracy_score(y_test_predict, y_test)
# training_score = clf.score(x_train, y_train)
# testing_score = clf.score(x_test, y_test)

# print("Training data score = {:.3f}".format(training_score))
# print("Testing data score = {:.3f}".format(testing_score))

# # **儲存模型**
# save_path = os.path.join(os.path.dirname(__file__), "save")
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)

# model_filename = 'RandomForest_CL_acc={:.3f}.pickle'.format(test_accuracy)
# with open(os.path.join(save_path, model_filename), 'wb') as f:
#     pickle.dump(clf, f)
    
import pickle
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# 讀取 log 資料
path = os.path.join(os.path.dirname(__file__), "..", "log")
allfile = os.listdir(path)
data_set = []

for file in allfile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))

# 特徵與標籤初始化
Ball_x, Ball_y, Vector_x, Vector_y, Direction, Platform, command = [], [], [], [], [], [], []

for data in data_set:
    for i in range(2, len(data["scene_info"]) - 2):
        sceneInfo = data["scene_info"][i]
        nextInfo = data["scene_info"][i + 1]
        cmd = data["command"][i]

        # 若不是 MOVE_LEFT、MOVE_RIGHT、NONE 就略過
        if cmd not in ["MOVE_LEFT", "MOVE_RIGHT", "NONE", ""]:
            continue

        # 特徵
        Ball_x.append(sceneInfo['ball'][0])
        Ball_y.append(sceneInfo['ball'][1])
        Platform.append(sceneInfo['platform'][0])
        dx = nextInfo["ball"][0] - sceneInfo["ball"][0]
        dy = nextInfo["ball"][1] - sceneInfo["ball"][1]
        Vector_x.append(dx)
        Vector_y.append(dy)

        if dx > 0:
            Direction.append(0 if dy > 0 else 1)
        else:
            Direction.append(2 if dy > 0 else 3)

        # 標籤（分類）
        if cmd in ["NONE", ""]:
            command.append(0)
        elif cmd == "MOVE_LEFT":
            command.append(-1)
        elif cmd == "MOVE_RIGHT":
            command.append(1)

# 將特徵與標籤轉為陣列
X = np.array(list(zip(Ball_x, Ball_y, Vector_x, Vector_y, Direction, Platform)))
Y = np.array(command)



# 分割資料
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 接收 PHP 傳入參數
php_max_depth = int(sys.argv[1]) 
php_random_state = int(sys.argv[2]) 
php_n_estimators = int(sys.argv[3])

# 建立分類模型
clf = RandomForestClassifier(max_depth=php_max_depth,
                             random_state=php_random_state,
                             n_estimators=php_n_estimators)
clf.fit(x_train, y_train)

# 驗證與測試
y_val_predict = clf.predict(x_val)
y_test_predict = clf.predict(x_test)

acc_val = accuracy_score(y_val, y_val_predict)
f1_val = f1_score(y_val, y_val_predict, average='weighted')

acc_test = accuracy_score(y_test, y_test_predict)
train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

print("Validation Accuracy: {:.3f}".format(acc_val))
print("Validation F1 Score: {:.3f}".format(f1_val))
print("Training Score: {:.3f}".format(train_score))
print("Testing Score: {:.3f}".format(test_score))

# 儲存模型
save_path = os.path.join(os.path.dirname(__file__), "save")
if not os.path.isdir(save_path):
    os.mkdir(save_path)

model_filename = 'RandomForest_CL_acc={:.2f}.pickle'.format(acc_test)
with open(os.path.join(save_path, model_filename), 'wb') as f:
    pickle.dump(clf, f)

