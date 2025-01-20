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

        # 正確載入儲存的機器學習模型
        model_path = os.path.join(
            os.path.dirname(__file__), 
            'save', 
            'KNN_regression_k=2_rmse=10.78_data=121722.pickle'
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

        # 如果球還未發出，進行發球操作
        if not scene_info["ball_served"]:
            self.ball_served = True  # 標記球已經發出
            self.previous_ball = scene_info['ball']  # 記錄當前球的位置
            command = "SERVE_TO_RIGHT"  # 發球指令：向右發球
        else:
            # 提取當前球的位置
            Ball_x = scene_info['ball'][0]
            Ball_y = scene_info['ball'][1]
            # 計算球的速度
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

            # 使用模型預測平台應移動到的位置
            x = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction]).reshape((1, -1))  # 構造輸入特徵
            y = self.model.predict(x)  # 使用模型進行預測

            # 根據預測位置決定平台移動指令
            if scene_info['platform'][0] + 20 - 5 < y:  # 平台中心偏左於預測位置
                command = "MOVE_RIGHT"  # 向右移動
            elif scene_info['platform'][0] + 20 - 5 > y:  # 平台中心偏右於預測位置
                command = "MOVE_LEFT"  # 向左移動
            else:
                command = "NONE"  # 平台已在正確位置，保持不動

        # 更新前一個球的位置為當前球的位置
        self.previous_ball = scene_info['ball']
        return command  # 返回指令

    def reset(self):
        """
        重置狀態，用於新一輪遊戲
        """
        self.ball_served = False  # 重置球的發出狀態
