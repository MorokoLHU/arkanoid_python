import pickle  # 用於序列化和反序列化資料
import numpy as np  # 用於數值運算
import os  # 用於操作系統相關功能（如路徑操作）
from sklearn.model_selection import train_test_split  # 用於分割訓練集和測試集
from sklearn.metrics import mean_squared_error  # 用於計算均方誤差
from math import sqrt  # 用於計算平方根
from sklearn.neighbors import KNeighborsRegressor  # KNN 回歸模型

# 設定資料路徑
path = os.path.join(os.path.dirname(__file__), '../../../log')  # 日誌資料夾路徑
files = os.listdir(path)  # 獲取資料夾中所有檔案名稱
data_set = []  # 用於儲存所有載入的資料

# 載入資料
for file in files:
    try:
        with open(os.path.join(path, file), 'rb') as f:  # 以二進制讀取檔案
            data = pickle.load(f)  # 反序列化資料
            print(f"{file} loaded successfully")  # 載入成功訊息
            data_set.append(data)  # 將資料添加到集合中
    except Exception as e:
        print(f"Error loading {file}: {e}")  # 載入失敗訊息

# 初始化資料結構
Ball_x = []  # 球的 x 座標
Ball_y = []  # 球的 y 座標
Speed_x = []  # 球的水平速度
Speed_y = []  # 球的垂直速度
Direction = []  # 球的移動方向
Command = []  # 玩家指令

# 解析資料
for data in data_set:
    for i, scene_info in enumerate(data['scene_info'][2:-3]):  # 遍歷每個場景資訊
        Ball_x.append(data['scene_info'][i+1]['ball'][0])  # 記錄球的 x 座標
        Ball_y.append(data['scene_info'][i+1]['ball'][1])  # 記錄球的 y 座標
        Speed_x.append(data['scene_info'][i+1]['ball'][0] - scene_info['ball'][0])  # 計算水平速度
        Speed_y.append(data['scene_info'][i+1]['ball'][1] - scene_info['ball'][1])  # 計算垂直速度
        # 判斷球的方向
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

    # 處理指令資料
    for command in data['command'][2:-2]:
        if command == 'NONE' or command == '':  # 未移動
            Command.append(0)
        elif command == 'MOVE_LEFT':  # 向左移動
            Command.append(-1)
        elif command == 'MOVE_RIGHT':  # 向右移動
            Command.append(1)
        else:  # 其他情況，移除最後一筆對應的資料
            Ball_x = Ball_x[:-1]
            Ball_y = Ball_y[:-1]
            Speed_x = Speed_x[:-1]
            Speed_y = Speed_y[:-1]
            Direction = Direction[:-1]

# 轉換 Command 為 numpy 陣列並調整形狀
Command = np.array(Command)
Command = Command.reshape(len(Command), 1)

# 組合特徵資料 X
X = np.array([0, 0, 0, 0, 0])  # 初始化 X
for i in range(len(Ball_x)):
    X = np.vstack((X, np.array([Ball_x[i], Ball_y[i], Speed_x[i], Speed_y[i], Direction[i]])))
X = X[1::]  # 移除初始化資料

# 計算預測目標位置
Position_pred = []  # 平台目標位置
platform_position_y = 400  # 平台的垂直位置
ball_seed_y = 7  # 球的移動速度
platform_width = 200  # 平台寬度
for i in range(len(Ball_x)):
    pred = Ball_x[i] + ((platform_position_y - Ball_y[i]) // ball_seed_y) * Speed_x[i]  # 預測位置
    section = (pred // platform_width)  # 計算區段
    if section % 2 == 0:  # 偶數區段
        pred = abs(pred - platform_width * section)
    else:  # 奇數區段
        pred = platform_width - abs(pred - platform_width * section)
    Position_pred.append(pred)

# 轉換為 numpy 陣列作為目標值 Y
Position_pred = np.array(Position_pred)
Y = Position_pred

# 資料總長度
length = len(Ball_x)

# KNN 模型訓練與驗證
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 資料分為訓練集與測試集（8:2）

# 初始化最佳參數
RMSE = 10000  # 初始 RMSE 設為很大值
k_final = 0  # 最佳鄰居數初始化

# 尋找最佳 k 值
for k in range(2, 50):
    model = KNeighborsRegressor(n_neighbors=k)  # 建立 KNN 回歸模型
    model.fit(x_train, y_train)  # 訓練模型

    # 評估模型
    y_predict = model.predict(x_test)  # 進行預測
    mse = mean_squared_error(y_test, y_predict)  # 計算均方誤差
    rmse = sqrt(mse)  # 計算 RMSE
    print("k =", k, ", RMSE = %.2f" % rmse)  # 印出 k 和 RMSE
    if rmse < RMSE:  # 更新最佳 RMSE 和 k 值
        RMSE = rmse
        k_final = k

# 使用最佳 k 值重新訓練模型
model = KNeighborsRegressor(n_neighbors=k_final)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
rmse = sqrt(mse) 
print("k =", k_final, "RMSE = %.2f" % rmse)  # 印出最佳 k 和對應的 RMSE

# 儲存模型
save_path = os.path.join(os.path.dirname(__file__), 'save')  # 模型儲存路徑
os.makedirs(save_path, exist_ok=True)  # 確保資料夾存在
model_filename = "KNN_regression_k={}_rmse={:.2f}_data={}.pickle".format(k_final, rmse, length)  # 檔案名稱
model_path = os.path.join(save_path, model_filename)

# 儲存模型到檔案
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
