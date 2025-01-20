"""
The template of the main script of the machine learning process
"""
import os
import pickle
from datetime import datetime
import random

class MLPlay:
    def __init__(self,ai_name, *args, **kwargs):
        """
        Constructor
        """
        print(ai_name)
        self.scene_info = []
        self.commands = []
        self.data = {'scene_info': [], 'command': []}

        self.ball_served = False
        self.previous_ball = (0, 0)
        self.pred = 100
        self.platform_y = 400
        self.ball_speed_y = 7
        self.platform_width = 200

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
            command = random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])
        else: # rule code
            self.pred = random.randint(0, 199)
            if self.previous_ball[1] - scene_info['ball'][1] > 0:
                pass    # ball is moving up
            else:
                distance_platform_ball_y = self.platform_y - scene_info['ball'][1]
                ball_speed_x = scene_info['ball'][0] - self.previous_ball[0]
                self.pred = scene_info['ball'][0] + ((distance_platform_ball_y // self.ball_speed_y) * ball_speed_x)

            section = (self.pred // self.platform_width)
            if section % 2 == 0:
                self.pred = abs(self.pred - self.platform_width * section)
            else:
                self.pred = self.platform_width - abs(self.pred - self.platform_width * section)

            if scene_info['platform'][0] + 20 - 5 < self.pred:
                command = "MOVE_RIGHT"
            elif scene_info['platform'][0] + 20 - 5 > self.pred:
                command = "MOVE_LEFT"
            else:
                command = "NONE"

        self.scene_info.append(scene_info)
        self.commands.append(command)

        self.previous_ball = scene_info['ball']
        return command

    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False

        self.data['scene_info'] = self.scene_info
        self.data['command'] = self.commands

        filepath = 'log/'
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        with open(os.path.join(os.path.dirname(__file__),
                               '../log/scen_info_{:%Y_%M_%d_%H_%M_%S}.pickle'.format(datetime.now())),
                  'wb') as f:
            pickle.dump(self.data, f)

        self.scene_info = []
        self.commands = []