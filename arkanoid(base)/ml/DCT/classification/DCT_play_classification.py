"""
The template of the main script of the machine learning process (classification version)
"""
import pickle  # 用於序列化和反序列化模型
import os  # 用於操作系統相關功能（如路徑操作）
import numpy as np  # 用於數值計算

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        初始化函數，用於設定機器學習玩家的初始狀態。
        """
        print(ai_name)  # 輸出 AI 名稱

        self.ball_served = False  # 標記球是否已發出
        self.previous_ball = (0, 0)  # 儲存上一次球的位置

        # 載入預先訓練的分類模型
        with open(os.path.join(os.path.dirname(__file__), 'save', 
            'DCT_classification_depth=47_acc=0.76_data=121748.pickle'), 'rb') as f:
            self.model = pickle.load(f)  # 反序列化模型並加載

    def update(self, scene_info, *args, **kwargs):
        """
        根據接收到的場景資訊生成指令。
        """
        # 如果遊戲結束或通關，返回 "RESET" 以通知重置遊戲狀態
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        # 如果球尚未發出，執行發球邏輯
        if not scene_info["ball_served"]:
            self.ball_served = True  # 設定球已發出標記
            self.previous_ball = scene_info['ball']  # 記錄球的初始位置
            command = "SERVE_TO_RIGHT"  # 發球指令向右
        else:
            # 提取球的當前位置
            Ball_x = scene_info['ball'][0]  # 球的 x 座標
            Ball_y = scene_info['ball'][1]  # 球的 y 座標

            # 計算球的速度
            Speed_x = scene_info['ball'][0] - self.previous_ball[0]  # 水平速度
            Speed_y = scene_info['ball'][1] - self.previous_ball[1]  # 垂直速度

            # 判斷球的運動方向
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

            # 提取平台的 x 座標
            Platform_x = scene_info['platform'][0]  # 平台的 x 座標

            # 將特徵組合成輸入向量（包含 Platform_x）
            x = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction, Platform_x]).reshape((1, -1))

            # 使用模型進行預測
            y = self.model.predict(x)[0]

            # 將預測結果轉換為對應的指令
            if y == 1:  # 類別 1 表示向右移動
                command = "MOVE_RIGHT"
            elif y == -1:  # 類別 -1 表示向左移動
                command = "MOVE_LEFT"
            else:  # 類別 0 表示保持不動
                command = "NONE"

        # 更新上一個球的位置
        self.previous_ball = scene_info['ball']
        return command  # 返回生成的指令

    def reset(self):
        """
        重置遊戲狀態，清除發球標記。
        """
        self.ball_served = False
        self.previous_ball = (0, 0)  # 重置前一個球的位置