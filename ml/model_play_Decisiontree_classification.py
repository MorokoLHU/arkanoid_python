"""
Playing game by KNN
"""

import pickle
import os
import numpy as np

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor
        """
        print(ai_name)

        self.ball_served = False
        self.previous_ball = (0, 0)
        modelname ='DCT_classification_depth=43_acc=0.70_data=94621.pickle'
        # 載取 model, 需要填入 model 名稱
        with open(os.path.join(os.path.dirname(__file__), "save", modelname), "rb") as f:
            self.model = pickle.load(f)

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_LEFT"
        else:
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Speed_x = scene_info["ball"][0] - self.previous_ball[0]
            Speed_y = scene_info["ball"][1] - self.previous_ball[1]
            Platform = scene_info["platform"][0]
            if Speed_x > 0:
                if Speed_y > 0: Direction = 0
                else :      	Direction = 1
            else:
                if Speed_y > 0: Direction = 2
                else :	        Direction = 3
            
            x = np.array([Ball_x,Ball_y,Speed_x, Speed_y, Direction, Platform]).reshape((1, -1))
            y = self.model.predict(x)
            if y == 0: command = "NONE"
            elif y == -1: command = "MOVE_LEFT"
            elif y == 1: command = "MOVE_RIGHT"
        
        self.previous_ball = scene_info["ball"]
        return command
    def reset(self):self.ball_served = False