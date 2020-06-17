'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/6/12
'''

import numpy as np
import random

def argmax(vector:list):

    cur_max = -np.inf
    max_index = []

    for i, item in enumerate(vector):
        if item > cur_max:
            cur_max = item
            max_index = []
            max_index.append(i)
        elif item == cur_max:
            max_index.append(i)

    return random.choice(max_index)


def softmax(vector:list):
    arr = np.array(vector)
    arr = np.exp(arr)/np.exp(arr).sum()

    return np.random.choice(list(range(4)),1,p=arr).tolist()[0]


def existing_node(Tree, obs):
    for item in Tree:
        if obs == item.obs:
            return True

    return False


def convert_key(obs):
    state = np.array(obs, dtype=np.str).flatten().tolist()

    return "".join(state)

