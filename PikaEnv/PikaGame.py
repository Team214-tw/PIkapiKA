import subprocess
import time
from multiprocessing import Pipe, Process
import math

import cv2
import mss
import numpy as np
from pynput.keyboard import Controller, Key
from transitions import Machine, State
from xvfbwrapper import Xvfb

from .PikaConst import ACTIONS, HEIGHT, WIDTH


def get_ball_position(img):
    return np.where((img[:, :, 0] == 191) & (img[:, :, 1] == 0) & (img[:, :, 2] == 191))


def get_leftPika_position(img):
    return np.where((img[:, : WIDTH // 2, 0] == 255) & (img[:, : WIDTH // 2, 1] == 0) & (img[:, : WIDTH // 2, 2] == 255))


def get_rightPika_position(img):
    return np.where((img[:, WIDTH // 2:, 0] == 255) & (img[:, WIDTH // 2:, 1] == 0) & (img[:, WIDTH // 2:, 2] == 255))


def pika_client(conn):
    monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}

    def press_and_release(keyboard, keys, holdtime=0.01):
        for key in keys:
            keyboard.press(key)
        time.sleep(holdtime)
        for key in keys:
            keyboard.release(key)

    def start_wine():
        subprocess.Popen([r"wine", r"PIKA_V.exe"])
        time.sleep(5)
        # get windows_id
        xdp = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--class", "wine"],
            stdout=subprocess.PIPE,
            check=True,
        )
        window_id = xdp.stdout
        # adjust window position
        cmd = ["xdotool", "windowmove", window_id, "0", "0"]
        subprocess.run(cmd, check=True)
        cmd = ["xdotool", "windowsize", window_id, str(WIDTH), str(HEIGHT)]
        subprocess.run(cmd, check=True)

    def start_game(keyboard, sct):
        # keep pressing enter until game start
        while True:
            img = np.array(sct.grab(monitor))
            rows, cols = get_ball_position(img)
            press_and_release(keyboard, keys=(Key.enter, ), holdtime=0.2)
            if len(rows) != 0 and len(cols) != 0 and np.sum(img == 0) <= 500000:
                press_and_release(keyboard, keys=(Key.alt, ))
                press_and_release(keyboard, keys=("g", ))
                press_and_release(keyboard, keys=("p", ))
                break
        img = np.array(sct.grab(monitor))
        conn.send(img)

    with Xvfb(width=WIDTH, height=WIDTH):
        # create instance in current desktop
        keyboard = Controller()
        sct = mss.mss()
        # initialize game
        start_wine()
        start_game(keyboard, sct)
        # wait actions
        while True:
            press_flow = conn.recv()
            for keys in press_flow:
                press_and_release(keyboard, keys=keys)
            img = np.array(sct.grab(monitor))
            conn.send(img)


class PikaGame(object):
    def _spawn_pika(self):
        parent_conn, child_conn = Pipe()
        p = Process(target=pika_client, args=(child_conn, ))
        p.start()
        return parent_conn, parent_conn.recv()

    def __init__(self):
        self.actions = ACTIONS
        self.last_ball_pos = None
        self.parent_conn, self.screenshot = self._spawn_pika()
        self.pika_init_positions = self._get_positions()[:4]
        self.FSM = self._init_FSM()
        print("-- client ready")

    def toggle_pause(self):
        self.parent_conn.send([(Key.alt,), ("g",), ("p",)])
        self.screenshot = self.parent_conn.recv()

    def take_action(self, idx):
        action = self.actions[idx]
        self.parent_conn.send([action])
        self.screenshot = self.parent_conn.recv()

    def observe(self):
        done = False
        result = self._get_positions()

        is_initial = np.array_equal(result[:4], self.pika_init_positions)

        if is_initial and self.FSM.is_begining():
            self.FSM.wait()

        if not is_initial and self.FSM.is_waiting():
            self.FSM.play()

        ball_pika_distance = math.sqrt((result[4] - result[2]) ** 2 + (result[5] - result[3]) ** 2)
        if result[4] >= HEIGHT * 0.8 and ball_pika_distance > 50 and self.FSM.is_gaming():
            self.FSM.end()
            done = True

        # in case that above condition failed
        if is_initial and self.FSM.is_gaming():
            self.FSM.end()
            done = True

        return result, done

    def _get_positions(self):
        result = np.zeros(shape=(8,), dtype=np.float32)
        while True:
            img = self.screenshot

            # left_pika
            rows, cols = get_leftPika_position(img)
            if len(rows) == 0 or len(cols) == 0:
                self._update_screenshot()
                continue

            result[0] = rows[0]
            result[1] = cols[0]

            # right_pika
            rows, cols = get_rightPika_position(img)
            if len(rows) == 0 or len(cols) == 0:
                self._update_screenshot()
                continue

            result[2] = rows[0]
            result[3] = cols[0] + WIDTH // 2

            # ball
            rows, cols = get_ball_position(img)
            if len(rows) == 0 or len(cols) == 0:  # when ball touch object
                self._update_screenshot()
                continue

            result[4] = rows[0]
            result[5] = cols[0]
            result[6] = self.last_ball_pos[0] if self.last_ball_pos else 0
            result[7] = self.last_ball_pos[1] if self.last_ball_pos else 0
            self.last_ball_pos = (result[4], result[5])

            return result

    def _update_screenshot(self):
        self.parent_conn.send([])
        self.screenshot = self.parent_conn.recv()

    def _init_FSM(self):
        states = [
            State(name='begining', on_enter=[], on_exit=[]),
            State(name='waiting', on_enter=[], on_exit=[]),
            State(name='gaming', on_enter=[], on_exit=[self._cleanup]),
        ]
        transitions = [
            {'trigger': 'wait', 'source': 'begining', 'dest': 'waiting'},
            {'trigger': 'play', 'source': 'waiting', 'dest': 'gaming'},
            {'trigger': 'end', 'source': 'gaming', 'dest': 'begining'},
        ]
        return PikaGameState(states, transitions, 'begining')

    def _cleanup(self):
        self.last_ball_pos = None


class PikaGameState(object):
    def __init__(self, states, transitions, initial_state):
        self.machine = Machine(model=self,
                               states=states,
                               transitions=transitions,
                               initial=initial_state)
