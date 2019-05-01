import numpy as np
import gym
from gym.spaces import Discrete
from contextlib import contextmanager
import time

class SnakeEnv(gym.Env):
    SIZE=100
  
    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space=Discrete(self.SIZE+1)
        self.action_space=Discrete(len(dices))

        for k,v in list(self.ladders.items()):
            self.ladders[v] = k
        self.pos = 1

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, a):
        step = np.random.randint(1, self.dices[a] + 1)
        self.pos += step
        if self.pos == 100:
            return 100, 100, 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        pass


class TableAgent(object):
    def __init__(self, num_ladders, env):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.n # 101

        self.r = [env.reward(s) for s in range(self.state_size)] # (101,)

        # stochastic policy: optimizing target
        #self.pi = np.zeros((self.state_size, self.action_size)) # \pi(a_t | s_t)
        self.pi = np.zeros(self.state_size, dtype=np.int) # 1D is because we assume a state corresponds to only one action

        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)
        
        # transition probability
        self.p = np.zeros((self.action_size, self.state_size, self.state_size), dtype=np.float) # p(s_{t+1} | s_t, a_t)
        for index, dice in enumerate(env.dices): # eg: env.dices: [3, 6]
            prob = 1.0 / dice # eg: 1/3 or 1/6
            for src in range(1, self.state_size - 1): # 0 and 100 should not be initialized
                #step = np.arange(1, dice + 1) # eg: [1, 2, 3] or [1, 2, 3, 4, 5, 6]
                step = np.arange(dice)
                dsts = src + step # a vector of dst
                for dst in dsts:
                    if dst > 100:
                        dst = 200 - dst # move back
                    dst = ladder_move(dst)
                    self.p[index, src, dst] += prob

        self.p[:, 100, 100] = 1 # If the src is 100, player can go nowhere but 100

        # state-value function
        self.value_s = np.zeros(self.state_size) # v_{\pi}(s)

        # action-value function
        self.value_sa = np.zeros((self.state_size, self.action_size)) # q_{\pi}(s, a)

        # discount factor
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]

class ModelFreeAgent(object):
    def __init__(self, env):
        # no num_ladders, since we assume that the transition probabilities are unknown
        
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        
        self.pi = np.zeros((self.state_size), dtype=np.int)
        self.value_sa = np.zeros((self.state_size, self.action_size), dtype=np.float)
        self.value_n = np.zeros((self.state_size, self.action_size), dtype=np.int) # N: accumulated quantity
        
        self.gamma = 0.8
    
    def play(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return self.pi[state]

def eval_game(env, policy):
    state = env.reset()
    return_val = 0
    while True:
        if isinstance(policy, TableAgent) or isinstance(policy, ModelFreeAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise Exception('Illegal policy')
        state, reward, terminate, _ = env.step(act)
        # print(state)
        return_val += reward
        if terminate:
            break
    return return_val



@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST:{}'.format(name, end - start))
