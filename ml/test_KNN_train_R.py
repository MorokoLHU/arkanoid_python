import sys
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor

path = os.path.join(os.path.dirname(__file__), '../log')
files = os.listdir(path)
data_set = []
for file in files:
    with open(os.path.join(path, file), 'rb') as f:
        data = pickle.load(f)
        data_set.append(data)

Ball_x = []
Ball_y = []
Speed_x = []
Speed_y = []
Direction = []
Command = []

for data in data_set:
    for i, scene_info in enumerate(data['scene_info'][2:-3]):
        Ball_x.append(data['scene_info'][i+1]['ball'][0])
        Ball_y.append(data['scene_info'][i+1]['ball'][1])
        Speed_x.append(data['scene_info'][i+1]['ball'][0] - scene_info['ball'][0])
        Speed_y.append(data['scene_info'][i+1]['ball'][1] - scene_info['ball'][1])
        if Speed_x[-1] > 0:
            if Speed_y[-1] > 0: Direction.append(0) # the ball is falling toward bottom-right direction
            else: Direction.append(1)               # the ball is rising toward top-right direction
        else:
            if Speed_y[-1] > 0: Direction.append(2) # the ball is falling toward bottom-left direction
            else: Direction.append(3)               # the ball is rising toward top-left direction

    for command in data['command'][2:-2]:
        if command == 'NONE' or command == '':
            Command.append(0)
        elif command == 'MOVE_LEFT':
            Command.append(-1)
        elif command == 'MOVE_RIGHT':
            Command.append(1)
        else:
            Ball_x = Ball_x[:-1]
            Ball_y = Ball_y[:-1]
            Speed_x = Speed_x[:-1]
            Speed_y = Speed_y[:-1]
            Direction = Direction[:-1]

Command = np.array(Command)
Command = Command.reshape(len(Command), 1)

X = np.array([0, 0, 0, 0, 0])
for i in range(len(Ball_x)):
    X = np.vstack((X, np.array([Ball_x[i], Ball_y[i], Speed_x[i], Speed_y[i], Direction[i]])))
X = X[1::]

Position_pred = []
platform_position_y = 400
ball_seed_y = 7
platform_width = 200
for i in range(len(Ball_x)):
    pred = Ball_x[i] + ((platform_position_y - Ball_y[i])//ball_seed_y) * Speed_x[i]

    section = (pred // platform_width)
    if section % 2 == 0:
        pred = abs(pred - platform_width * section)
    else:
        pred = platform_width - abs(pred - platform_width * section)

    Position_pred.append(pred)

Position_pred = np.array(Position_pred)
Y = Position_pred

length = len(Ball_x)

# training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


k_value = int(sys.argv[1])
# k_range = range(1,30)
# best_k = 1
# best_rmse = float("inf")

# for k in k_range:
#     model = KNeighborsRegressor(n_neighbors = k)
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
    
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = sqrt(mse)
    
#     print(f"k = {k}, RMSE = {rmse:.3f}")
    
#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_k = k

model = KNeighborsRegressor(n_neighbors=k_value)
model.fit(x_train, y_train) # 擬合 training

# evaluation
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
training_score = model.score(x_train, y_train)
testing_score = model.score(x_test, y_test)

# print(f"Best k = {best_k}, Final RMSE = {rmse:.3f}")
# print(f"Training score = {training_score:.3f}")
# print(f"Testing score = {testing_score:.3f}")

filepath = os.path.join(os.path.dirname(__file__), 'save')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

with open(os.path.join(filepath, 'KNN_regression_k={}_rmse={:.3f}_data={}.pickle'.format(best_k,rmse,len(X))), 'wb') as f:
    pickle.dump(model, f)