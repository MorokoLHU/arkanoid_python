import pickle  # 用於序列化和反序列化 Python 物件
import os  # 用於操作系統相關功能（如路徑操作）
import numpy as np  # 用於數值運算的工具庫

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        初始化函式，用於設置初始狀態和載入模型
        """
        print(ai_name)  # 印出 AI 的名稱

        self.ball_served = False  # 標記球是否已經發出
        self.previous_ball = (0, 0)  # 初始化記錄球的前一個位置

        # 載入事先訓練好的 KNN 分類模型
        model_path = os.path.join(
            os.path.dirname(__file__), 
            'save', 
            'KNN_classification_k=2_accuracy=74.50%_data=121748.pickle'  # 替換為實際的模型檔案名稱
        )  # 獲取模型的路徑
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)  # 載入模型

    def update(self, scene_info, *args, **kwargs):
        """
        根據接收到的遊戲畫面資訊生成對應的指令
        """
        # 如果遊戲狀態為結束或通關，返回 "RESET" 指令，重新開始遊戲
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"
        
        command = "NONE"  # 默認不移動

        # 如果球還未發出，進行發球操作
        if not scene_info["ball_served"]:
            self.ball_served = True  # 標記球已經發出
            self.previous_ball = scene_info['ball']  # 記錄當前球的位置
            command = "SERVE_TO_RIGHT"  # 發球指令：向右發球
        else:
            # 提取當前球的位置
            Ball_x = scene_info['ball'][0]  # 當前球的 x 座標
            Ball_y = scene_info['ball'][1]  # 當前球的 y 座標

            # 計算球的速度（x 和 y 的變化量）
            Speed_x = scene_info['ball'][0] - self.previous_ball[0]
            Speed_y = scene_info['ball'][1] - self.previous_ball[1]

            # 根據速度計算球的移動方向
            if Speed_x > 0:
                if Speed_y > 0: 
                    Direction = 0  # 球向右下移動
                else: 
                    Direction = 1  # 球向右上移動
            else:
                if Speed_y > 0: 
                    Direction = 2  # 球向左下移動
                else: 
                    Direction = 3  # 球向左上移動

            Platform = scene_info['platform'][0]  # 平台的 x 座標

            # 構造輸入特徵向量
            x = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction, Platform]).reshape((1, -1))  # 確保包含 6 個特徵
            # 使用分類模型預測下一步應執行的玩家指令
            y = self.model.predict(x)  # 預測結果（0: NONE, 1: MOVE_LEFT, 2: MOVE_RIGHT）
            y = int(y[0])  # 提取單一預測值

            # 將預測結果轉換為對應的指令
            if y == 0:
                command = "NONE"  # 不移動
            elif y == 1:
                command = "MOVE_LEFT"  # 向左移動
            elif y == 2:
                command = "MOVE_RIGHT"  # 向右移動

        # 更新前一個球的位置為當前球的位置
        self.previous_ball = scene_info['ball']
        return command  # 返回生成的指令

    def reset(self):
        """
        重置狀態，用於新一輪遊戲
        """
        self.ball_served = False  # 重置球的發出狀態
        self.previous_ball = (0, 0)  # 重置前一個球的位置
