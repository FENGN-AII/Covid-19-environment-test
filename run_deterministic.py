import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np

import virl


def deterministic_train():
    pass
    #deterministic agent need no train

def run_deterministic(stochastic=False, noisy=False):
    if stochastic:
        episodes=1
    else:
        episodes=10

    for problem in range(0,episodes):
        #Setting environment parameters
        if stochastic:
            env = virl.Epidemic(stochastic=True, noisy=noisy)
        else:
            env = virl.Epidemic(stochastic=stochastic, noisy=noisy, problem_id=problem)
        for act in range(4):
            env.reset()
            rewards = []
            done = False
            while not done:
                s, r, done, i = env.step(action=act) # deterministic agent
                rewards.append(r)
            if stochastic:
                print('Stochastic=Ture, with Noisy=' + str(noisy) + ', with action ' + str(act) + ', rewards=' + str(sum(rewards)))
            else:
                print('Problem ' + str(problem) + ' with action ' + str(act) + ' reward sum: ' + str(sum(rewards)))
