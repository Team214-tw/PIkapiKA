import argparse
import os
import pickle
import time

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from APEX.model import DQN
from APEX.prioritized_memory import Memory

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--actor-num',
                    type=int,
                    default=0,
                    help='number of actors')
args = parser.parse_args()

args.device = 'cpu'


class Learner:
    def __init__(self,
                 state_size,
                 action_size,
                 update_period=100,
                 batch_size=64,
                 gamma=0.99,
                 actor_num=1,
                 mem_size=10_000,
                 learning_rate=0.001,
                 chkpt_dir='log/0'):

        self.state_size = state_size
        self.action_size = action_size
        self.update_period = update_period
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_num = actor_num
        self.mem_size = mem_size
        self.lr = learning_rate
        self.chkpt_dir = chkpt_dir
        os.makedirs(chkpt_dir, exist_ok=True)

        # create evaluate model and target model
        self.eval_model = DQN(state_size, action_size).to(args.device)
        self.target_model = DQN(state_size, action_size).to(args.device)
        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=self.lr)
        self.memory = Memory(self.mem_size)
        self.train_cntr = 0

    def main(self):
        self.save_model()
        while True:
            for i in range(self.actor_num):
                self.load_memory(i)

            self.train_model()

            if (self.train_cntr // 10000) % args.actor_num == 0:
                self.save_model()

    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        if self.memory.tree.n_entries % self.update_period == 0:
            self.update_target_model()

        print('train')

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()
        indices = np.arange(self.batch_size)

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4].astype(int)

        states = T.FloatTensor(states).to(args.device)
        actions = T.LongTensor(actions).to(args.device)
        rewards = T.FloatTensor(rewards).to(args.device)
        next_states = T.FloatTensor(next_states).to(args.device)
        dones = T.LongTensor(dones).to(args.device)
        is_weights = T.FloatTensor(is_weights).to(args.device)

        # Q function of current state
        V_s, A_s = self.eval_model.forward(states)
        q_pred = T.add(V_s,
                       (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        # Q function of next state
        V_s_, A_s_ = self.target_model.forward(next_states)
        q_next = T.add(V_s_,
                       (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        # Q Learning: get maximum Q value at s' from target model
        q_target = rewards + (1 - dones) * self.gamma * q_next.max(dim=1)[0]

        # update priority using error
        errors = T.abs(q_pred - q_target).data.cpu().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (is_weights * F.mse_loss(q_pred, q_target)).mean()
        loss.backward()

        # and train
        self.optimizer.step()

    # after some time interval update the target model to be same with model

    def update_target_model(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())

    def load_memory(self, actor_id):
        filepath = os.path.join(self.chkpt_dir, f'memory{actor_id}.pt')
        self.memory.load(filepath)
        print(f'Load memory from {actor_id}, memory size: {self.memory.tree.n_entries}')

    def save_model(self):
        model = {'eval': self.eval_model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'train_cntr': self.train_cntr,
                 }
        filepath = os.path.join(self.chkpt_dir, 'model.pt')
        with open(filepath, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        filepath = os.path.join(self.chkpt_dir, 'model.pt')
        model = T.load(filepath)
        self.train_cntr = model['train_cntr']
        self.eval_model.load_state_dict(model['eval'])
        self.optimizer.load_state_dict(model['optimizer'])
        print('Learner: model loaded from', filepath)


if __name__ == '__main__':
    learner = Learner(state_size=8,
                      action_size=11,
                      actor_num=args.actor_num,
                      chkpt_dir='log/pika')
    learner.main()
