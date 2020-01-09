import math
import gym
from time import sleep
from gym import spaces
import numpy as np
from cv2 import cv2 as cv
from .PikaGame import PikaGame
from .PikaConst import WIDTH, HEIGHT


class PikaEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        super(PikaEnv, self).__init__()
        #  [left_pika_row, left_pika_col
        #  right_pika_row, right_pika_col,
        #  ball_row, ball_col,
        #  last_ball_row, last_ball_col]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([HEIGHT, WIDTH, HEIGHT, WIDTH,
                           HEIGHT, WIDTH, HEIGHT, WIDTH]),
            dtype=np.float32,
        )
        self.reward_range = (-math.inf, math.inf)
        self.action_space = spaces.Discrete(11)
        self.game = PikaGame()

    def _calc_reward(self, obs):
        def touch(pikax, pikay, ballx, bally, last_ballx, last_bally):
            pika_bx_l = int(pikax - 5)   # vertical line: x = pika_bx_l
            pika_bx_h = int(min(pikax + 50, WIDTH))  # vertical line: x = pika_bx_h
            pika_by_l = int(pikay - 20)  # horizontal line: y = pika_by_l
            pika_by_h = int(pikay + 40)  # horizontal line: y = pika_by_h
            ball_bx_l = int(pika_bx_l - 40)
            ball_bx_h = int(min(pika_bx_h + 40, WIDTH))
            ball_by_l = int(pika_by_l - 40)
            ball_by_h = int(pika_by_h)
            # try:
            #     a = np.zeros((ball_by_h-ball_by_l, ball_bx_h-ball_bx_l, 3))
            #     a.fill(255)
            #     self.game.screenshot[ball_by_l:ball_by_h, ball_bx_l:ball_bx_h, :3] = a
            #     self.game.screenshot[pika_by_l:pika_by_h, pika_bx_l:pika_bx_h, :3] = np.zeros(
            #         (pika_by_h-pika_by_l, pika_bx_h-pika_bx_l, 3))
            # except:
            #     pass
            if ballx > ball_bx_l and ballx < ball_bx_h and bally > ball_by_l and bally < ball_by_h:
                # calculate ball line
                dx = last_ballx - ballx
                dy = last_bally - bally
                if dx == 0:  # vertical line: x = ballx
                    return True if ballx < pika_bx_h and ballx > pika_bx_l else False
                elif dy == 0:  # horizontal line: y = bally
                    return True if bally < pika_by_h and bally > pika_by_l else False
                else:  # line: y = ax + b
                    a = dy / dx
                    b = bally - a * ballx
                    # check pika_bx_l
                    y = a * pika_bx_l + b
                    if y < pika_by_h and y > pika_by_l:
                        return True
                    # check pika_bx_h
                    y = a * pika_bx_h + b
                    if y < pika_by_h and y > pika_by_l:
                        return True
                    # check pika_by_l
                    x = (pika_by_l - b) / a
                    if x < pika_bx_h and x > pika_bx_l:
                        return True
                    # check pika_by_h
                    x = (pika_by_h - b) / a
                    if x < pika_bx_h and x > pika_bx_l:
                        return True
            return False

        # goal
        reward = 0
        pikay = obs[2]
        pikax = obs[3]
        bally = obs[4]
        ballx = obs[5]
        last_bally = obs[6]
        last_ballx = obs[7]

        if bally >= HEIGHT * 0.8:
            dx = ballx - pikax
            dy = bally - pikay
            # --- pika try to rescue ball
            # if math.sqrt(dx ** 2 + dy ** 2) < 40:
            #     reward += 50
            if ballx > WIDTH // 2:  # pika lose
                reward -= 10
            else:  # pika win
                reward += 10

        # --- stand in the middle
        # if abs(pikax - WIDTH * 0.75) < 35:  
        #     reward += 5
        # elif abs(pikax - WIDTH * 0.5) < 60 or abs(pikax - WIDTH) < 60: # stand front or back
        #     reward -= 5
        # else:
        #     reward -= 3

        # --- pika touch ball
        # if touch(pikax, pikay, ballx, bally, last_ballx, last_bally):
        #     reward += 10

        return reward

    def step(self, action):
        self.game.take_action(action)
        obs, done = self.game.observe()
        reward = self._calc_reward(obs)
        if done:
            self.game.toggle_pause()
        return obs, reward, done, {}

    def reset(self):
        self.game.toggle_pause()
        return np.random.random_sample((8,))

    def render(self, mode="human"):
        cv.imshow("PIKA game", self.game.screenshot)
        if cv.waitKey(25) & 0xFF == ord("q"):
            cv.destroyAllWindows()
