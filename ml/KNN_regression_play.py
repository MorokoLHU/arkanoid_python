import pickle
import csv
import os
import numpy as np
class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):

        print(ai_name)
        
        self.model_name = "LinearRegression_RE_rmse=60.03.pickle"
        
        self.result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
        os.makedirs(self.result_dir,exist_ok = True)
        self.result_path = os.path.join(self.result_dir, "model_name.csv")
        
        
        self.ball_served = False
        self.previous_ball = (0, 0)

        # 載取 model, 需要更換 model 名稱
        with open(os.path.join(os.path.dirname(__file__), 'save',self.model_name ), 'rb') as f:
            self.model = pickle.load(f)
    def update(self, scene_info, *args, **kwargs):

        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or scene_info["status"] == "GAME_PASS"):
            self.save_model_name()
            return "RESET"
        
        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_RIGHT"
        else:
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Speed_x = scene_info["ball"][0] - self.previous_ball[0]
            Speed_y = scene_info["ball"][1] - self.previous_ball[1]
            
            if Speed_x > 0:
                if Speed_y > 0:
                    Direction = 0  # 球往右下
                else:
                    Direction = 1  # 球往右上
            else:
                if Speed_y > 0:
                    Direction = 2  # 球往左下
                else:
                    Direction = 3  # 球往左上
            
            x = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction]).reshape((1, -1))  # 展開成一列
            y = self.model.predict(x)  # 使用model預測Ball X落點
            
            if scene_info["platform"][0] + 20 + 5 < y:
                command = "MOVE_RIGHT"
            elif scene_info["platform"][0] + 20 - 5 > y:
                command = "MOVE_LEFT"
            else:
                command = "NONE"
        
        self.previous_ball = scene_info["ball"]
        return command

    def reset(self):self.ball_served = False
    def get_model_info(self):
        return{
            "model_name": self.model_name 
        }

    def save_model_name(self):
        model_name_without_extension = os.path.splitext(self.model_name)[0]
        result_data = {            
            "model_name": model_name_without_extension
            }

        file_exist = os.path.isfile(self.result_path)
        with open(self.result_path, mode = 'a', newline= '') as f:
            writer = csv.DictWriter(f, fieldnames = result_data.keys())
            if not file_exist:
                writer.writeheader()
            writer.writerow(result_data)