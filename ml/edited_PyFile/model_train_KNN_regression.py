import pickle
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor

path = os.path.join(os.path.dirname(__file__), "..", "log")
allFile = os.listdir(path)
data_set = []

for file in allFile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))
        
Ball_x = []
Ball_y = []
Speed_x = []
Speed_y = []
Direction = []

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

# Feature
X = np.array([0, 0, 0, 0, 0])
X = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction]).T
Position_pred = [] #預測ball_x落點位置
platform_position_y = 400
ball_speed_y = 7
platform_width = 200
for i in range(len(Ball_x)):
    pred = Ball_x[i] +  ((platform_position_y - Ball_y[i])//ball_speed_y) * Speed_x[i]
    
    section = (pred // platform_width)
    if(section % 2 == 0):
        pred = abs(pred - platform_width*section)
    else:
        pred = platform_width - abs(pred - platform_width*section)
    
    Position_pred.append(pred)
    
Position_pred = np.array(Position_pred)
Y = Position_pred

length = len(Ball_x)


#### KNN ###
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 資料拆成8:2的訓練及測試

RMSE = 10000
k_final = int(sys.argv[1]) 
#使其接收PHP的參數

# training in best k
model = KNeighborsRegressor(n_neighbors=k_final)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
rmse = sqrt(mse)
print("k =", k_final, "RMSE = %.2f" % rmse)

# save the model
with open(os.path.join(os.path.dirname(__file__), 'save',
                       "KNN_regression_k={}_rmse={:.2f}_data={}.pickle".format(k_final, rmse, length)), 'wb') as f:
    pickle.dump(model, f)
