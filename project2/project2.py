import sys
import pandas as pd
import numpy as np
import time

class QLearning:
    def __init__(self, n_states, n_actions, discount, lr):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_rate = discount # gamma
        self.Q = np.zeros((n_states, n_actions)) # action value function Q(s, a)
        self.Q.fill(-np.inf) 
        self.lr = lr # alpha: learning rate

def update(m, s, a, r, sp):
    if m.Q[s,a] == -np.inf:
        m.Q[s,a] = 0.
    m.Q[s, a] += m.lr * (r + m.discount_rate * np.max(m.Q[sp]) - m.Q[s, a])
    return m

if __name__ == '__main__':
    tic = time.perf_counter() 
    dataset = sys.argv[1]
    df = pd.read_csv(f'data/{dataset}.csv')
    df['s'] -= 1
    df['a'] -= 1
    df['sp'] -= 1
    if dataset == 'medium':
        model = QLearning(50000, df['a'].max()+1, 1., 0.01)
    else:
        model = QLearning(df['s'].max()+1, df['a'].max()+1, 0.95, 0.01)
    for i, obs in df.iterrows():
        model = update(model, obs['s'], obs['a'], obs['r'], obs['sp'])
    toc = time.perf_counter()
    print('Elapsed time:', toc-tic, 'seconds')
    policy = np.argmax(model.Q, axis=1) + 1
    np.savetxt(f'data/{dataset}.policy', policy.astype(int), fmt='%i')