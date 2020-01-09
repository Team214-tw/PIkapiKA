import argparse
import os
import pickle
import sys
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from APEX.model import DQN
from APEX.prioritized_memory import Memory
from PikaEnv.PikaEnv import PikaEnv

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--actor-id',
                    type=int,
                    default=0,
                    help='actor id')
parser.add_argument('--render',
                    action='store_true',
                    default=False,
                    help='actor id')
parser.add_argument('--benchmark',
                    action='store_true',
                    default=False,
                    help='benchmark for project used')
parser.add_argument('--cuda',
                    action='store_true',
                    default=False,
                    help='enable cuda')
args = parser.parse_args()

args.device = "cuda:0" if args.cuda else "cpu"


class Actor:
    def __init__(self,
                 env,
                 state_size,
                 action_size,
                 batch_size=64,
                 gamma=0.99,
                 epsilon=0.9,
                 eps_dec=0.9999,
                 eps_min=0.05,
                 mem_size=100_000,
                 actor_id=0,
                 chkpt_dir='log/0'):

        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.mem_size = mem_size
        self.actor_id = actor_id
        self.chkpt_dir = chkpt_dir
        os.makedirs(chkpt_dir, exist_ok=True)

        self.model = DQN(state_size=state_size,
                         action_size=action_size).to(args.device)
        self.memory = Memory(self.mem_size)
        self.explore_step = 1000
        self.explore_cntr = 0

    def main(self):
        self.load_model()
        scores = []
        avg_scores = []

        self.total_time = 0
        self.record_cntr = 0
        self.record = []
        for episode in range(100):
            score = self.round(episode)
            scores.append(score)
            self.benchmark(episode, scores)
            self.save_memory()
            self.load_model()
            print("\nepisode:", episode, "  score:", score, "  memory length:",
                  self.memory.tree.n_entries, "  epsilon:", self.epsilon)

            if args.benchmark:
                print(self.record)
                if len(self.record) == 10:
                    break

        if args.benchmark:
            print(np.mean(self.record))

    def benchmark(self, episode, scores):
        filepath = os.path.join(
            self.chkpt_dir, f'benchmark{self.actor_id}.png')
        # plt.ylim(top=500)
        plt.plot(scores, color='blue')
        plt.savefig(filepath)

    def round(self, episode):
        state = env.reset()
        state = np.reshape(state, [1, self.state_size])

        done = False
        score = 0

        while not done:
            start_time = time.time()
            if args.render:
                self.env.render()

            action = self.get_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, self.state_size])

            if env.game.FSM.is_gaming() or done:
                # print(reward)
                score += reward
                self.append_sample(state, action, reward, next_state, done)

                if args.benchmark:
                    self.total_time += time.time() - start_time
                    if self.total_time > 5:
                        self.record.append(self.record_cntr)
                        self.record_cntr = 0
                        self.total_time = 0
                    self.record_cntr += 1

            if done:
                break

            state = next_state

            if self.explore_cntr >= self.explore_step:
                self.epsilon_decay()
            else:
                self.explore_cntr += 1

        return score

    def epsilon_decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

    def append_sample(self, state, action, reward, next_state, done):
        state_tensor = T.FloatTensor(state).to(args.device)
        V_s, A_s = self.model.forward(state_tensor)
        old_val = A_s.data[0][action]

        if done:
            new_val = reward
        else:
            new_state = T.FloatTensor(next_state).to(args.device)
            V_s_, A_s_ = self.model.forward(new_state)
            new_val = reward + self.gamma * T.max(A_s_.data)

        error = abs(old_val - new_val)

        error = error.cpu()

        self.memory.add(error, (state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.random() > self.epsilon:  # choose best action
            state_tensor = T.FloatTensor(state).to(args.device)
            _, advantage = self.model.forward(state_tensor)
            action = T.argmax(advantage).item()
        else:  # random select action
            action = np.random.choice(self.action_size)

        return action

    def load_model(self):
        filepath = os.path.join(self.chkpt_dir, 'model.pt')
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                self.model.load_state_dict(model['eval'])
                print(f'Actor {self.actor_id}: model loaded from', filepath)
            except:
                pass
        else:
            print(f'Actor {self.actor_id}: no model found at', filepath)

    def save_memory(self):
        filepath = os.path.join(self.chkpt_dir, f'memory{self.actor_id}.pt')
        self.memory.save(filepath)
        self.memory.clear()


if __name__ == '__main__':
    env = PikaEnv()

    actor = Actor(env=env,
                  state_size=env.observation_space.shape[0],
                  action_size=env.action_space.n,
                  actor_id=args.actor_id,
                  chkpt_dir='log/pika')
    actor.main()
