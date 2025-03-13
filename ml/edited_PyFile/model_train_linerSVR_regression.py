import os
import pickle
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 加載數據
path = os.path.join(os.path.dirname(__file__), "..", "log")
allFile = os.listdir(path)
data_set = []

for file in allFile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))

# 數據處理
Ball_x, Ball_y, Speed_x, Speed_y, Direction = [], [], [], [], []

for data in data_set:
    for i, scene_info in enumerate(data["scene_info"][2:-3]):
        Ball_x.append(data["scene_info"][i + 1]["ball"][0])
        Ball_y.append(data["scene_info"][i + 1]["ball"][1])
        Speed_x.append(data["scene_info"][i + 1]["ball"][0] - data["scene_info"][i]["ball"][0])
        Speed_y.append(data["scene_info"][i + 1]["ball"][1] - data["scene_info"][i]["ball"][1])

        if Speed_x[-1] > 0:
            if Speed_y[-1] > 0:
                Direction.append(0)  # 向右下
            else:
                Direction.append(1)  # 向右上
        else:
            if Speed_y[-1] > 0:
                Direction.append(2)  # 向左下
            else:
                Direction.append(3)  # 向左上

# 構造特徵和標籤（使用 NumPy）
X = np.array([0, 0, 0, 0, 0])
for i in range(len(Ball_x)):
    X = np.vstack((X, np.array([Ball_x[i], Ball_y[i], Speed_x[i], Speed_y[i], Direction[i]])))
X = X[1::]

Position_pred = []
platform_position_y = 400
ball_speed_y = 7
platform_width = 200
for i in range(len(Ball_x)):
    pred = Ball_x[i] + ((platform_position_y - Ball_y[i])//ball_speed_y) * Speed_x[i]
    
    section = (pred // platform_width)
    if (section % 2 == 0):
        pred = abs(pred - platform_width * section)
    else:
        pred = platform_width - abs(pred - platform_width * section)

    Position_pred.append(pred)

Position_pred = np.array(Position_pred)
Y = Position_pred

length = len(Ball_x)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 構建管道模型


php_C = int(sys.argv[1])  # 接收PHP參數
php_epsilon = int(sys.argv[2])
regr = make_pipeline(StandardScaler(),LinearSVR(epsilon=php_epsilon, tol=0.0001, C=php_C, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual='auto', verbose=0, random_state=None, max_iter=1000))
regr.fit(X,Y)
# 訓練模型


# 預測並評估
y_pred = regr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print("RMSE = {:.2f}".format(rmse))

# 儲存模型
save_path = os.path.join(os.path.dirname(__file__), 'save')
os.makedirs(save_path, exist_ok=True)

model_file = os.path.join(save_path, "SVR_regression_C=1.0_rmse={:.2f}_data={}.pickle".format(rmse,length))
with open(model_file, 'wb') as f:
    pickle.dump(regr, f)

print(f"Model saved to {model_file}")
