#CL
import time
import pickle
import csv
import os
import numpy as np

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        print(ai_name)
        self.model_name = "SVM_CL_C=1.0.pickle"
        self.result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
        os.makedirs(self.result_dir, exist_ok=True)
        self.result_path = os.path.join(self.result_dir, "model_name.csv")
        self.Nowresult_path = os.path.join(self.result_dir, "Nowresult.csv")
        self.ball_served = False
        self.previous_ball = (0, 0)

        self.start_time = time.time()
        with open(os.path.join(os.path.dirname(__file__), "save", self.model_name), "rb") as f:
            self.model = pickle.load(f)

    def update(self, scene_info, *args, **kwargs):
        # 當遊戲結束或遊戲通過時，要求調用 `reset()` 以開始新的一輪
        if scene_info["status"] == "GAME_OVER" or scene_info["status"] == "GAME_PASS":
            self.save_model_name()
            return "RESET"
        if scene_info["frame"] >= 1000:
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_RIGHT"
        else:
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Speed_x = scene_info["ball"][0] - self.previous_ball[0]
            Speed_y = scene_info["ball"][1] - self.previous_ball[1]
            Platform = scene_info["platform"][0]
            if Speed_x > 0:
                Direction = 0 if Speed_y > 0 else 1
            else:
                Direction = 2 if Speed_y > 0 else 3

            x = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction, Platform]).reshape(1, -1)
            y = self.model.predict(x)
            if y == 0:
                command = "NONE"
            elif y == -1:
                command = "MOVE_LEFT"
            elif y == 1:
                command = "MOVE_RIGHT"
        
        self.previous_ball = scene_info["ball"]
        return command

    def reset(self):
        self.ball_served = False

    def get_model_info(self):
        return {
            "model_name": self.model_name
        }

    def save_model_name(self):
        model_name_without_extension = os.path.splitext(self.model_name)[0]
        result_data = {
            "model_name": model_name_without_extension
        }
        file_exist = os.path.isfile(self.result_path)
        with open(self.result_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result_data.keys())
            if not file_exist:
                writer.writeheader()
            writer.writerow(result_data)