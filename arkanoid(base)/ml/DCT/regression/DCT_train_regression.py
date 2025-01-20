import pickle  # 用於序列化和反序列化資料
import numpy as np  # 用於數值計算
import os  # 用於操作系統相關功能（如路徑操作）
from sklearn.model_selection import train_test_split  # 用於將資料分為訓練集與測試集
from sklearn.metrics import mean_squared_error  # 用於計算均方誤差
from math import sqrt  # 用於計算平方根
from sklearn.tree import DecisionTreeRegressor  # 引入決策樹回歸模型

# 定義資料的路徑
path = os.path.join(os.path.dirname(__file__), '../../../log')
files = os.listdir(path)  # 獲取目錄下的所有檔案
data_set = []  # 初始化資料集

# 載入資料
for file in files:
    try:
        with open(os.path.join(path, file), 'rb') as f:  # 以二進制模式讀取檔案
            data = pickle.load(f)  # 反序列化資料
            print(f"{file} loaded successfully")  # 打印成功訊息
            data_set.append(data)  # 將資料添加到資料集中
    except Exception as e:
        print(f"Error loading {file}: {e}")  # 如果發生錯誤，打印錯誤訊息

# 初始化特徵和目標變數的列表
Ball_x = []  # 儲存球的 x 座標
Ball_y = []  # 儲存球的 y 座標
Speed_x = []  # 儲存球的水平速度
Speed_y = []  # 儲存球的垂直速度
Direction = []  # 儲存球的方向
Command = []  # 儲存命令對應的編碼

# 從資料集中提取特徵
for data in data_set:
    for i, scene_info in enumerate(data['scene_info'][2:-3]):
        # 提取球的當前位置及速度
        Ball_x.append(data['scene_info'][i+1]['ball'][0])
        Ball_y.append(data['scene_info'][i+1]['ball'][1])
        Speed_x.append(data['scene_info'][i+1]['ball'][0] - scene_info['ball'][0])
        Speed_y.append(data['scene_info'][i+1]['ball'][1] - scene_info['ball'][1])

        # 判斷球的運動方向
        if Speed_x[-1] > 0:
            if Speed_y[-1] > 0:
                Direction.append(0)  # 球向右下移動
            else:
                Direction.append(1)  # 球向右上移動
        else:
            if Speed_y[-1] > 0:
                Direction.append(2)  # 球向左下移動
            else:
                Direction.append(3)  # 球向左上移動

    # 提取命令並進行編碼
    for command in data['command'][2:-2]:
        if command == 'NONE' or command == '':
            Command.append(0)  # 無動作
        elif command == 'MOVE_LEFT':
            Command.append(-1)  # 向左移動
        elif command == 'MOVE_RIGHT':
            Command.append(1)  # 向右移動
        else:
            # 如果命令無效，刪除對應的特徵資料
            Ball_x = Ball_x[:-1]
            Ball_y = Ball_y[:-1]
            Speed_x = Speed_x[:-1]
            Speed_y = Speed_y[:-1]
            Direction = Direction[:-1]

# 將命令轉換為 numpy 陣列並調整形狀
Command = np.array(Command).reshape(len(Command), 1)

# 建立特徵矩陣 X
X = np.array([0, 0, 0, 0, 0])  # 初始化特徵矩陣
for i in range(len(Ball_x)):
    X = np.vstack((X, np.array([Ball_x[i], Ball_y[i], Speed_x[i], Speed_y[i], Direction[i]])))
X = X[1:]  # 刪除初始化的第一行

# 計算平台應到達的位置
Position_pred = []  # 預測的平台位置
platform_position_y = 400  # 平台的垂直位置
ball_seed_y = 7  # 球的垂直移動速度
platform_width = 200  # 平台的寬度

for i in range(len(Ball_x)):
    # 根據球的位置和速度計算平台的目標位置
    pred = Ball_x[i] + ((platform_position_y - Ball_y[i]) // ball_seed_y) * Speed_x[i]

    # 處理碰撞後的反彈情況
    section = (pred // platform_width)
    if section % 2 == 0:
        pred = abs(pred - platform_width * section)
    else:
        pred = platform_width - abs(pred - platform_width * section)

    Position_pred.append(pred)

# 將目標變數轉為 numpy 陣列
Position_pred = np.array(Position_pred)
Y = Position_pred  # 設定 Y 為平台應移動的位置

# 訓練資料長度
length = len(Ball_x)

# 資料分割：80% 為訓練集，20% 為測試集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 初始化決策樹模型
model = DecisionTreeRegressor(criterion='squared_error', max_depth=30, splitter='best')
model.fit(x_train, y_train)  # 擬合模型

# 模型評估
y_pred = model.predict(x_test)  # 預測測試集
mse = mean_squared_error(y_test, y_pred)  # 計算均方誤差
rmse = sqrt(mse)  # 計算均方根誤差
print('rmse: ', rmse)  # 打印 RMSE
depth = model.tree_.max_depth  # 獲取樹的最大深度

# 儲存模型
filepath = os.path.join(os.path.dirname(__file__), 'save')
if not os.path.isdir(filepath):
    os.mkdir(filepath)  # 如果目錄不存在則創建

# 儲存模型至檔案
with open(os.path.join(filepath, f'DCT_regression_depth={depth}_rmse={rmse}.pickle'), 'wb') as f:
    pickle.dump(model, f)  # 將模型序列化並儲存
