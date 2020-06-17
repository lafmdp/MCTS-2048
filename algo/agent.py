'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/6/12
'''

import math
from utils.utils import argmax, convert_key
import random
import numpy as np

C = 100
gamma = 0.9

def compute_uct(N,Q):
    uct_list = []

    Ns = sum(N)
    if Ns == 0:
        return [np.inf, np.inf, np.inf, np.inf]

    for a in range(4):
        if N[a] == 0:
            uct_list.append(np.inf)
        else:
            uct_list.append(Q[a]+C*math.sqrt(math.log(Ns)/N[a]))

    return uct_list


class MCTSAgent():
    def __init__(self, env, deepth=100):

        self.env = env
        self.deepth = deepth

        self.tree = set()
        self.Ntable = {}
        self.Qtable = {}

    def rollout(self, obs, score, deepth):
        if deepth == 0:# or done:
            return 0

        a = random.randint(0,3)
        self.env.set(obs, score)
        new_obs, r, done, _ = self.env.step(a)
        return r + gamma * self.rollout(new_obs, self.env.score, deepth-1)


    def reset(self):
        self.Qtable = {}
        self.Ntable = {}
        self.tree = set()


    def get_action(self, obs, score):
        self.reset()

        for i in range(20):
            # self.reset()
            self.search(obs, score, self.deepth)

        return argmax(self.Qtable[convert_key(obs)])


    def search(self, obs, score, deepth:int):
        if deepth == 0:# or done:
            return 0

        converted_obs = convert_key(obs)

        if  converted_obs not in self.tree:

            self.tree.add(converted_obs)
            self.Qtable[converted_obs] = [0,0,0,0]
            self.Ntable[converted_obs] = [0,0,0,0]

            return self.rollout(obs, score, deepth)

        action = argmax(compute_uct(self.Ntable[converted_obs],
                                    self.Qtable[converted_obs]))

        self.env.set(obs, score)
        new_obs, r, done,_ = self.env.step(action)
        q = r + gamma * self.search(new_obs, self.env.score, deepth-1)

        self.Ntable[converted_obs][action] += 1
        self.Qtable[converted_obs][action] = self.Qtable[converted_obs][action] + (q-self.Qtable[converted_obs][action])/self.Ntable[converted_obs][action]

        return q

