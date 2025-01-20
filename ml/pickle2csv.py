import pickle
import os
import numpy as np
import csv


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
    print(data)
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

    for command in data['command'][2:-3]:
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
Y1 = Position_pred

Y2 = Command[:, 0]

print('len(X):', len(X))
print('len(Y1):', len(Y1))
print('len(Y2):', len(Y2))

# 開啟輸出的 CSV 檔案
with open('dataset.csv', 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    # 寫入一列資料
    writer.writerow(['ball_x', 'ball_y', 'speed_x', 'speed_y', 'direction', 'platform_x', 'command'])
    for idx, x in enumerate(X):
        writer.writerow([x[0], x[1], x[2], x[3], x[4], Y1[idx], Y2[idx]])

