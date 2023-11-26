import kins
import numpy as np
import matplotlib.pyplot as plt

num_lines = 2
max_len = 10
params = [.5, 0]
KinsEnv = kins.KinsEnv(num_lines, max_len, params)
sarsa = kins.SarsaAgent(KinsEnv, alpha=.1, gamma=.9, epsilon=.05, n=1, episode_length=50)

s = np.array([0, 0])
for i in range(10):
    r,s,t = KinsEnv.step(s, 0)
    print(r,s,t)
