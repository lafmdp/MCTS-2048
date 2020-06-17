'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/6/12
'''

import time
from algo.agent import MCTSAgent
from utils.game2048 import Game2048Env

if __name__ == '__main__':
    env = Game2048Env(True)
    env.seed(1)
    agent = MCTSAgent(Game2048Env(False), 200)

    obs = env.reset()
    done = False
    while not done:

        action = agent.get_action(obs, env.score)
        # time.sleep(5)
        obs, rew, done, info = env.step(action)

        if done:
            print(rew, done, info)
            time.sleep(10)


    env.close()

