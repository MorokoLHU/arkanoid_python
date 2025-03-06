
import pickle
import os
import numpy as np

class MLPlay:
    def __init__(self,ai_name, *args, **kwargs):
        """
        Constructor
        """
        print(ai_name)

        self.ball_served = False
        self.previous_ball = (0, 0)
        modelname = 'LinearRegression_rmse=58.85_data=94578.pickle'
        with open(os.path.join(os.path.dirname(__file__), 'save', modelname), 'rb') as f:
            self.model = pickle.load(f)

    def update(self, scene_info, *args, **kwargs):
        """
        Generate the command according to the received `scene_info`.
        """
        # Make the caller to invoke `reset()` for the next round.
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not scene_info["ball_served"]:
            self.ball_served = True
            self.previous_ball = scene_info['ball']
            command = "SERVE_TO_RIGHT"
        else:
            Ball_x = scene_info['ball'][0]
            Ball_y = scene_info['ball'][1]
            Speed_x = scene_info['ball'][0] - self.previous_ball[0]
            Speed_y = scene_info['ball'][1] - self.previous_ball[1]

            if Speed_x > 0:
                if Speed_y > 0: Direction = 0   # the ball is falling toward bottom-right direction
                else: Direction = 1             # the ball is rising toward top-right direction
            else:
                if Speed_y > 0: Direction = 2   # the ball is falling toward bottom-left direction
                else: Direction = 3             # the ball is rising toward top-left direction

            x = np.array([Ball_x, Ball_y, Speed_x, Speed_y, Direction]).reshape((1, -1))
            y = self.model.predict(x)

            if scene_info['platform'][0] + 20 - 5 < y:
                command = "MOVE_RIGHT"
            elif scene_info['platform'][0] + 20 - 5 > y:
                command = "MOVE_LEFT"
            else:
                command = "NONE"

        self.previous_ball = scene_info['ball']
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False
