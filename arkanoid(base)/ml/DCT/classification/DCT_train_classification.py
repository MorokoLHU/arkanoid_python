import pickle  # 用於序列化和反序列化資料
import numpy as np  # 用於數值計算
import os  # 用於操作系統相關功能（如路徑操作）
from sklearn.model_selection import train_test_split  # 用於將資料分為訓練集與測試集
from sklearn.metrics import accuracy_score, f1_score  # 用於計算分類準確率
from sklearn.tree import DecisionTreeClassifier  # 引入決策樹分類模型
from sklearn import tree

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
Platform = []  # 儲存平台的 x 座標
Command = []  # 儲存命令對應的編碼

# 從資料集中提取特徵
for data in data_set:
    for i, scene_info in enumerate(data['scene_info'][2:-2]):
        # 提取球的當前位置及速度
        Ball_x.append(scene_info['ball'][0])
        Ball_y.append(scene_info['ball'][1])
        Platform.append(scene_info['platform'][0])
        Speed_x.append(data['scene_info'][i + 1]['ball'][0] - data['scene_info'][i]['ball'][0])
        Speed_y.append(data['scene_info'][i + 1]['ball'][1] - data['scene_info'][i]['ball'][1])

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
    for command in data['command'][2:-2]:  # 對應的命令
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
            Platform = Platform[:-1]
            Speed_x = Speed_x[:-1]
            Speed_y = Speed_y[:-1]
            Direction = Direction[:-1]

# 確保 `X` 和 `Y` 對應一致
X = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction, Platform]).T
Y = np.array(Command)

print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# 建立特徵矩陣 X
X = np.array([0, 0, 0, 0, 0 ,0])  # 初始化特徵矩陣
for i in range(len(Ball_x)):
    X = np.vstack((X, [Ball_x[i], Ball_y[i], Speed_x[i], Speed_y[i], Direction[i], Platform[i]]))
X = X[1::]  # 刪除初始化的第一行

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)  # 資料拆成8:2的訓練及測試

model = tree.DecisionTreeClassifier()
model = model.fit(x_train,y_train)
y_predict = model.predict(x_test)
Accuracy = float('{:.3}'.format(accuracy_score(y_predict,y_test)))
depth = model.tree_.max_depth
print("depth = ", depth, "Accuracy = ", Accuracy)

FIScore = float('{:3f}'.format(f1_score(y_test, y_predict, average='weighted')))
print("F1 score=", FIScore)

# 儲存模型
path = os.path.join(os.path.dirname(__file__), 'save')
if not os.path.isdir(path):
    os.mkdir(path)  # 如果目錄不存在則創建

# 儲存模型至檔案
with open(os.path.join(os.path.dirname(__file__),'save',\
    'DCT_classification_depth={}_acc={:.2f}_data={}.pickle'.format(depth, Accuracy, len(X))), 'wb') as f:
    pickle.dump(model, f)  # 將模型序列化並儲存
